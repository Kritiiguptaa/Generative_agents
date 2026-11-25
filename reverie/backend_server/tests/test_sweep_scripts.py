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


def _script(path, body):
    with open(path, "w") as f:
        f.write("#!/bin/sh\n" + body)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)


@unittest.skipUnless(HAVE_BASH, "needs bash")
class SweepStatusRunning(unittest.TestCase):
    """A stage log with no copy is 'running' while the run is alive, 'stranded' after."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = os.path.join(self.tmp.name, "out")
        self.stage = os.path.join(self.tmp.name, "stage")
        os.makedirs(self.out)
        os.makedirs(self.stage)
        self.procs = []
        self.addCleanup(self._kill)

    def _kill(self):
        for p in self.procs:
            p.kill()
            p.wait()

    def _stage_log(self, name, age_s):
        path = os.path.join(self.stage, name + ".log")
        with open(path, "w") as f:
            f.write("partial\n")
        t = time.time() - age_s
        os.utime(path, (t, t))

    def _status(self):
        env = dict(os.environ, OUT=self.out, STAGE=self.stage, RUN="t1", SEEDS="42")
        r = subprocess.run(["bash", os.path.join(BACKEND, "sweep_status.sh")],
                           capture_output=True, text=True, env=env)
        return r.stdout

    def _fake_run(self, sim):
        p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)",
                              "--sim", sim, "--steps", "5"])
        self.procs.append(p)
        time.sleep(0.3)

    def _line(self, out, name):
        return next(l for l in out.splitlines() if l.strip().startswith(name))

    def test_a_freshly_written_stage_log_is_running(self):
        self._stage_log("mw_s42", age_s=5)
        out = self._status()
        self.assertIn("running", self._line(out, "mw_s42"))
        self.assertIn("running   : 1", out)
        self.assertIn("Do not start run_sweep.sh again", out)

    def test_an_old_stage_log_with_no_process_is_stranded(self):
        self._stage_log("mw_s42", age_s=3600)
        out = self._status()
        self.assertIn("stranded", self._line(out, "mw_s42"))
        self.assertIn("running   : 0", out)

    @unittest.skipUnless(shutil.which("pgrep"), "needs pgrep")
    def test_a_live_process_for_the_sim_is_running_even_with_a_quiet_log(self):
        self._stage_log("mw_s42", age_s=3600)
        self._fake_run("t1_mw_s42")
        self.assertIn("running", self._line(self._status(), "mw_s42"))

    @unittest.skipUnless(shutil.which("pgrep"), "needs pgrep")
    def test_a_process_for_a_longer_sim_name_does_not_match(self):
        # mw_s420 is not mw_s42
        self._stage_log("mw_s42", age_s=3600)
        self._fake_run("t1_mw_s420")
        self.assertIn("stranded", self._line(self._status(), "mw_s42"))


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
        self.assertIn(f"START mw_s42  (live log: {self.stage}/mw_s42.log)", r.stdout)
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

    def test_force_resume_overrides_the_guard(self):
        self._run()
        r = self._run(FORK="base_trading", FORCE_RESUME="1")
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn("FORCE_RESUME=1 set", r.stdout)


class CommitRecordedAtRunStart(unittest.TestCase):
    """A checkout made mid-run must not change the commit the report records."""

    def setUp(self):
        from middleware import metrics_recorder as m
        self.m = m
        m.reset_metadata_caches()
        self.addCleanup(m.reset_metadata_caches)

    @staticmethod
    def _git_says(commit):
        def fake(args, **kw):
            out = commit if args[1] == "rev-parse" else ""
            return subprocess.CompletedProcess(args, 0, stdout=out + "\n", stderr="")
        return mock.patch("middleware.metrics_recorder.subprocess.run", side_effect=fake)

    def test_the_commit_read_at_start_survives_a_later_checkout(self):
        with self._git_says("aaaaaaa"):
            self.m.prime_run_metadata()
        with self._git_says("bbbbbbb"):                      # someone checked out mid-run
            self.assertEqual(self.m.git_commit_info()["git_commit"], "aaaaaaa")
            self.assertEqual(self.m.collect_run_metadata(include_ollama_version=False)["git_commit"],
                             "aaaaaaa")

    def test_without_priming_the_commit_is_read_late(self):
        # The old behaviour this guards against: first use, at report time.
        with self._git_says("bbbbbbb"):
            self.assertEqual(self.m.git_commit_info()["git_commit"], "bbbbbbb")

    def test_run_primes_the_commit_before_the_first_step(self):
        # Source-structure check: the call must precede the step loop in
        # TradingReverie.run. PairedEvaluation inherits run(), so this covers both.
        with open(os.path.join(BACKEND, "trading_reverie.py")) as f:
            tree = ast.parse(f.read())
        run = next(n for c in tree.body if isinstance(c, ast.ClassDef) and c.name == "TradingReverie"
                   for n in c.body if isinstance(n, ast.FunctionDef) and n.name == "run")
        prime = loop = None
        for node in ast.walk(run):
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "prime_run_metadata":
                prime = node.lineno if prime is None else min(prime, node.lineno)
            if isinstance(node, ast.For) and loop is None:
                loop = node.lineno
        self.assertIsNotNone(prime, "run() never calls prime_run_metadata()")
        self.assertLess(prime, loop, "the commit must be read before the step loop starts")
        with open(os.path.join(BACKEND, "paired_eval.py")) as f:
            self.assertNotIn("def run(", f.read(), "paired_eval overrides run(); prime it there too")


if __name__ == "__main__":
    unittest.main()
