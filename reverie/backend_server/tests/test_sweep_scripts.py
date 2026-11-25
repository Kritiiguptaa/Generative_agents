"""run_sweep.sh, sweep_status.sh, and the commit recorded at run start.

The shell tests run the real scripts against temp folders with a fake `python`
and `curl` on PATH, so the sweep's control flow (START lines, .done markers,
SWEEP_META, the resume guard) is exercised without a model or a GPU.
"""
import ast
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
STORAGE = os.path.join(BACKEND, "..", "..", "environment", "frontend_server", "storage")
sys.path.insert(0, BACKEND)

HAVE_BASH = shutil.which("bash") is not None


def _install_fake_pgrep(shim_dir, procs_file):
    """
    A pgrep that reads pretend processes from `procs_file` (one command line per
    line) instead of the real process table. The real one made these tests depend
    on what else was running: on the shared DGX another person's sweep matched the
    script's "a trading_reverie.py is running" guard and failed the suite. The
    pattern is an ERE matched against the pretend lines, the same semantics the
    scripts rely on from pgrep -f, so the anchoring (mw_s42 vs mw_s420) is still
    tested. The last argument is the pattern; -a and -f are accepted and ignored.
    """
    _script(os.path.join(shim_dir, "pgrep"),
            'pat=""; for a in "$@"; do pat="$a"; done\n'
            f'[ -f "{procs_file}" ] || exit 1\n'
            f'grep -E -- "$pat" "{procs_file}" && exit 0\n'
            'exit 1\n')


def _script(path, body):
    with open(path, "w") as f:
        f.write("#!/bin/sh\n" + body)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)


@unittest.skipUnless(HAVE_BASH, "needs bash")
class SweepStatusRunning(unittest.TestCase):
    """A staged log with no copy: stranded if it holds a report, running if a process has it, else killed."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = os.path.join(self.tmp.name, "out")
        self.stage = os.path.join(self.tmp.name, "stage")
        os.makedirs(self.out)
        os.makedirs(self.stage)
        self.shim = os.path.join(self.tmp.name, "shim")
        os.makedirs(self.shim)
        self.procs_file = os.path.join(self.tmp.name, "procs.txt")
        _install_fake_pgrep(self.shim, self.procs_file)

    def _stage_log(self, name, finished=False):
        path = os.path.join(self.stage, name + ".log")
        with open(path, "w") as f:
            f.write("per-step output\n" + ("HALLUCINATION REPORT\n" if finished else ""))

    def _status(self):
        env = dict(os.environ, PATH=self.shim + os.pathsep + os.environ["PATH"],
                   OUT=self.out, STAGE=self.stage, RUN="t1", SEEDS="42")
        r = subprocess.run(["bash", os.path.join(BACKEND, "sweep_status.sh")],
                           capture_output=True, text=True, env=env)
        return r.stdout

    def _fake_run(self, sim):
        with open(self.procs_file, "a") as f:
            f.write(f"4242 python -u trading_reverie.py --fork x --sim {sim} --steps 5\n")

    def _line(self, out, name):
        return next(l for l in out.splitlines() if l.strip().startswith(name))

    def test_a_staged_log_with_a_report_is_stranded(self):
        # the run finished and wrote its report; only the copy to OUT failed
        self._stage_log("mw_s42", finished=True)
        self.assertIn("stranded", self._line(self._status(), "mw_s42"))

    def test_a_staged_log_with_no_report_and_a_live_process_is_running(self):
        self._stage_log("mw_s42")
        self._fake_run("t1_mw_s42")
        out = self._status()
        self.assertIn("running", self._line(out, "mw_s42"))
        self.assertIn("running   : 1", out)
        self.assertIn("Do not start run_sweep.sh again", out)

    def test_a_staged_log_with_no_report_and_no_process_is_killed(self):
        self._stage_log("mw_s42")
        out = self._status()
        self.assertIn("killed", self._line(out, "mw_s42"))
        self.assertIn("running   : 0", out)

    def test_a_process_for_a_longer_sim_name_does_not_match(self):
        # mw_s420 is not mw_s42
        self._stage_log("mw_s42")
        self._fake_run("t1_mw_s420")
        self.assertIn("killed", self._line(self._status(), "mw_s42"))


@unittest.skipUnless(HAVE_BASH, "needs bash")
class RunSweepEndToEnd(unittest.TestCase):
    """The real run_sweep.sh with a fake python and a fake Ollama."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.shim = os.path.join(self.tmp.name, "shim")
        os.makedirs(self.shim)
        _script(os.path.join(self.shim, "curl"),
                'echo \'{"models":[{"name":"mistral:latest"},{"name":"nomic-embed-text:latest"}]}\'\n')
        _script(os.path.join(self.shim, "python"), 'echo "HALLUCINATION REPORT"\n')
        self.out = os.path.join(self.tmp.name, "out")
        self.stage = os.path.join(self.tmp.name, "stage")

    def _run(self, **extra):
        env = dict(os.environ,
                   PATH=self.shim + os.pathsep + os.environ["PATH"],
                   OUT=self.out, STAGE=self.stage, SIM_STORAGE=STORAGE,
                   SEEDS="42", STEPS="1", MIN_FREE_MB="1", WRITE_TEST_MB="1",
                   MAX_HOURS="1", RUN="t1", FORK="base_trading_3agent")
        env.update(extra)
        return subprocess.run(["bash", os.path.join(BACKEND, "run_sweep.sh")],
                              capture_output=True, text=True, env=env, cwd=BACKEND)

    def test_start_lines_say_where_to_watch_and_configs_are_marked_done(self):
        r = self._run()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn(f"START mw_s42   (live: tail -f {self.stage}/mw_s42.log)", r.stdout)
        for name in ("mw_s42", "base_s42", "pmw_s42", "pbase_s42"):
            self.assertTrue(os.path.exists(os.path.join(self.out, name + ".done")), name)

    def test_the_sweep_records_what_it_ran_and_a_resume_skips_finished_configs(self):
        self._run()
        with open(os.path.join(self.out, "SWEEP_META")) as f:
            meta = f.read()
        self.assertIn("fork=base_trading_3agent", meta)
        self.assertIn("roster=Alex Chen,Marcus Webb,Sara Kim", meta)
        again = self._run()
        self.assertEqual(again.returncode, 0)
        self.assertIn("SKIP  mw_s42", again.stdout)

    def test_a_resume_with_a_different_roster_is_refused(self):
        self._run()
        r = self._run(FORK="base_trading")
        self.assertEqual(r.returncode, 1)
        self.assertIn("was started with different settings", r.stdout)

    def test_arms_limits_which_configs_run(self):
        r = self._run(ARMS="mw,base")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        for name in ("mw_s42", "base_s42"):
            self.assertTrue(os.path.exists(os.path.join(self.out, name + ".done")), name)
        for name in ("pmw_s42", "pbase_s42"):
            self.assertFalse(os.path.exists(os.path.join(self.out, name + ".done")), name)
        self.assertIn("arms: mw base", r.stdout)
        self.assertIn("seed 42  COMPLETE", r.stdout)             # complete for the requested arms

    def test_sweep_status_follows_arms_and_does_not_report_skipped_arms_missing(self):
        self._run(ARMS="mw,base")
        env = dict(os.environ, OUT=self.out, STAGE=self.stage, RUN="t1", SEEDS="42", ARMS="mw,base")
        out = subprocess.run(["bash", os.path.join(BACKEND, "sweep_status.sh")],
                             capture_output=True, text=True, env=env).stdout
        self.assertIn("All requested configs are complete", out)
        self.assertNotIn("pmw_s42", out)

    def test_an_unknown_arm_is_refused_before_anything_runs(self):
        r = self._run(ARMS="mw,nope")
        self.assertEqual(r.returncode, 2)
        self.assertIn("unknown arm 'nope'", r.stdout)
        self.assertFalse(os.path.exists(os.path.join(self.out, "mw_s42.done")))

    def test_force_resume_overrides_the_guard(self):
        self._run()
        r = self._run(FORK="base_trading", FORCE_RESUME="1")
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn("FORCE_RESUME=1 set", r.stdout)


@unittest.skipUnless(HAVE_BASH, "needs bash")
class CompareForksSmoke(unittest.TestCase):
    """compare_forks_smoke.sh with a fake python: one run per fork and seed, no overlap."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.shim = os.path.join(self.tmp.name, "shim")
        os.makedirs(self.shim)
        self.calls = os.path.join(self.tmp.name, "calls.txt")
        self.procs_file = os.path.join(self.tmp.name, "procs.txt")
        _install_fake_pgrep(self.shim, self.procs_file)
        _script(os.path.join(self.shim, "python"),
                f'echo "$@" >> {self.calls}\necho "HALLUCINATION REPORT"\n')
        self.out = os.path.join(self.tmp.name, "cmp")

    def _run(self, **extra):
        env = dict(os.environ, PATH=self.shim + os.pathsep + os.environ["PATH"],
                   OUT=self.out, FORKS="base_trading base_trading_active", SEEDS="42 43", STEPS="3")
        env.update(extra)
        return subprocess.run(["bash", os.path.join(BACKEND, "compare_forks_smoke.sh")],
                              capture_output=True, text=True, env=env, cwd=BACKEND)

    def test_runs_every_fork_and_seed_with_the_same_settings_and_named_logs(self):
        r = self._run()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        with open(self.calls) as f:
            calls = f.read().splitlines()
        self.assertEqual(len(calls), 4)
        for fork in ("base_trading", "base_trading_active"):
            for seed in ("42", "43"):
                self.assertTrue(os.path.exists(os.path.join(self.out, fork, f"mw_s{seed}.log")))
                self.assertTrue(any(f"--fork {fork} " in c and f"--seed {seed} " in c and
                                    f"--sim cmp_{fork}_mw_s{seed} " in c for c in calls), (fork, seed))
        self.assertTrue(all("--steps 3 " in c and "--temperature 0.7 " in c and "--fresh" in c for c in calls))

    def test_refuses_to_start_while_a_trading_reverie_process_is_running(self):
        with open(self.procs_file, "w") as f:
            f.write("30424 python -u trading_reverie.py --fork base_trading --sim run07_base_s42\n")
        r = self._run()
        self.assertEqual(r.returncode, 1)
        self.assertIn("already running", r.stdout)
        self.assertIn("run07_base_s42", r.stdout)            # names the process in the way
        self.assertFalse(os.path.exists(self.calls))



if __name__ == "__main__":
    unittest.main()
