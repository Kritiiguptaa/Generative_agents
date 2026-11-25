/* ===========================================================================
   ANCHOR-1 -- the handheld console on the landing page.

   A pixel trading floor that replays a real run. Nothing here is a canned
   loop: the agents standing at the desks are the run's personas on the tiles
   it recorded for them, and every order ticket that flies to the market board
   is a decision out of trading_log.json. Tickets the middleware flagged are
   intercepted in mid-air and stamped BLOCKED -- which is the one claim this
   whole project makes, drawn rather than asserted.

   Three layers, in order:

     1. sprite compositor  agents are built pixel by pixel into offscreen
                           canvases once at boot, then blitted. Cheap per
                           frame, and it means an agent's look is derived
                           from their name rather than shipped as an asset.
     2. scene              the room, in logical 352x240 pixels, supersampled
                           so canvas text stays crisp while sprites stay
                           chunky.
     3. replay             a clock that walks the timeline and spawns tickets.

   Everything is laid out in logical pixels. The canvas backing store is
   scaled by SS; no coordinate in this file is ever in device pixels.
   =========================================================================== */
(function () {
  "use strict";

  var payload = document.getElementById("consoleData");
  var cv = document.getElementById("floorCanvas");
  if (!payload || !cv || !cv.getContext) return;

  var RUNS = JSON.parse(payload.textContent).filter(function (r) {
    return r.console.agents.length;
  });
  if (!RUNS.length) return;

  var ctx = cv.getContext("2d");
  var reduced = window.matchMedia &&
                window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---- palette ---------------------------------------------------------- */

  var INK = "#1A1320", CREAM = "#FFF8E7";
  var GREEN = "#6BCF7F", RED = "#FF6B6B", TEAL = "#4ECDC4", GOLD = "#FFD93D";

  /* Agent colours are assigned by seat, not by name, so a run always reads
     left-to-right in the same order as the tab strip beneath the card. */
  var SEAT_COL = [
    { shirt: [255, 202, 84],  hair: [58, 42, 28],   tag: GOLD },
    { shirt: [78, 205, 196],  hair: [92, 60, 34],   tag: TEAL },
    { shirt: [255, 107, 107], hair: [120, 76, 42],  tag: RED },
    { shirt: [107, 207, 127], hair: [64, 48, 28],   tag: GREEN }
  ];

  /* =========================================================================
     1. sprite compositor
     ========================================================================= */

  var SW = 16, SH = 30;                 // one agent sprite, in logical pixels
  var OUTLINE = [38, 34, 46];
  var SKIN = { hi: [255, 221, 189], base: [247, 201, 170], sh: [212, 158, 126], line: [168, 112, 82] };

  var bw, bh;                           // buffer dims for the pixel helpers
  function clamp(v) { return v < 0 ? 0 : v > 255 ? 255 : Math.round(v); }
  function shades(c) {
    return [[clamp(c[0] * 1.22), clamp(c[1] * 1.22), clamp(c[2] * 1.22)],
            [c[0], c[1], c[2]],
            [clamp(c[0] * 0.68), clamp(c[1] * 0.68), clamp(c[2] * 0.68)]];
  }
  function put(b, x, y, c, a) {
    if (x < 0 || x >= bw || y < 0 || y >= bh) return;
    var i = (y * bw + x) * 4;
    b[i] = c[0]; b[i + 1] = c[1]; b[i + 2] = c[2];
    b[i + 3] = a === undefined ? 255 : a;
  }
  function alphaAt(b, x, y) {
    if (x < 0 || x >= bw || y < 0 || y >= bh) return 0;
    return b[(y * bw + x) * 4 + 3];
  }
  function rect(b, x0, y0, x1, y1, c) {
    for (var y = y0; y <= y1; y++) for (var x = x0; x <= x1; x++) put(b, x, y, c);
  }

  var HX0 = 3, HX1 = 12;                // head spans these columns

  function drawHead(b) {
    for (var y = 3; y <= 14; y++) for (var x = HX0; x <= HX1; x++) {
      // knock the four corners off so the head reads round, not as a box
      if ((x === HX0 || x === HX1) && (y === 3 || y === 14)) continue;
      put(b, x, y, SKIN.base);
    }
    for (y = 5; y < 11; y++) put(b, HX0 + 1, y, SKIN.hi);      // lit side
    for (y = 5; y < 13; y++) put(b, HX1 - 1, y, SKIN.sh);      // shaded side
    [HX0 - 1, HX1 + 1].forEach(function (ex) {                 // ears
      put(b, ex, 8, SKIN.base); put(b, ex, 9, SKIN.sh);
    });
    rect(b, 6, 15, 9, 16, SKIN.sh);                            // neck
  }

  function drawFace(b, mood) {
    var white = [250, 248, 244], pupil = [46, 38, 42];
    [5, 10].forEach(function (x) { put(b, x, 8, white); put(b, x + 1, 8, pupil); });
    if (mood === "flat")  [5, 6, 10, 11].forEach(function (x) { put(b, x, 6, SKIN.line); });
    if (mood === "sharp") { put(b, 5, 7, SKIN.line); put(b, 6, 6, SKIN.line);
                            put(b, 10, 6, SKIN.line); put(b, 11, 7, SKIN.line); }
    if (mood === "soft")  { [5, 11].forEach(function (x) { put(b, x, 6, SKIN.line); });
                            [6, 10].forEach(function (x) { put(b, x, 6, SKIN.sh); }); }
    put(b, 8, 10, SKIN.sh); put(b, 7, 11, SKIN.sh);            // nose
    var mouth = [158, 86, 80];
    [6, 7, 8, 9].forEach(function (x) { put(b, x, 12, mouth); });
    if (mood === "soft") { put(b, 5, 11, mouth); put(b, 10, 11, mouth); }
  }

  /* Three hair silhouettes. Which one an agent gets is picked from their seat,
     so the three on a floor never collide. */
  function drawHair(b, style, colour) {
    var s = shades(colour), hi = s[0], base = s[1], sh = s[2], x, y;
    rect(b, HX0, 1, HX1, 3, base);
    rect(b, HX0 - 1, 2, HX1 + 1, 4, base);
    for (y = 4; y < 7; y++) { put(b, HX0 - 1, y, base); put(b, HX0, y, base);
                              put(b, HX1, y, base); put(b, HX1 + 1, y, base); }
    if (style === 1) {                                  // side part
      for (y = 1; y < 5; y++) put(b, 6, y, sh);
      for (x = HX0; x < 6; x++) put(b, x, 2, hi);
    } else if (style === 2) {                           // floppy fringe
      for (x = 5; x <= 11; x++) put(b, x, 5, base);
      [8, 9, 10].forEach(function (fx) { put(b, fx, 6, base); });
      [6, 7, 8].forEach(function (fx) { put(b, fx, 1, hi); });
    } else {                                            // long, framing
      for (y = 5; y <= 14; y++) {
        put(b, HX0 - 2, y, base); put(b, HX0 - 1, y, base);
        put(b, HX1 + 1, y, base); put(b, HX1 + 2, y, base);
      }
      for (x = HX0; x < 8; x++) put(b, x, 1, hi);
    }
  }

  function drawTorso(b, colour, back) {
    var s = shades(colour), base = s[1], sh = s[2], y;
    rect(b, 4, 17, 11, 17, base);
    rect(b, 3, 18, 12, 23, base);
    for (y = 18; y <= 23; y++) { put(b, 3, y, sh); put(b, 12, y, sh); }
    if (back) { for (y = 18; y <= 23; y++) put(b, 7, y, sh); return; }
    var collar = [238, 238, 236];
    [[7, 17], [8, 17], [6, 18], [9, 18]].forEach(function (p) { put(b, p[0], p[1], collar); });
    for (y = 19; y <= 23; y++) { put(b, 7, y, sh); put(b, 8, y, sh); }   // placket
  }

  /* phase 0 = both feet down, 1 and 2 = alternating steps. */
  function drawLegs(b, phase) {
    var trouser = [54, 56, 70], shoe = [44, 40, 48];
    [[4, 6], [9, 11]].forEach(function (leg, i) {
      rect(b, leg[0], 24, leg[1], 27, trouser);
      var lifted = (phase === 1 && i === 0) || (phase === 2 && i === 1);
      rect(b, leg[0], lifted ? 27 : 28, leg[1], lifted ? 27 : 28, shoe);
    });
  }

  /* One black pixel around every filled edge. Cheap, and it is what makes the
     sprites read at this size against a busy floor. */
  function outline(b) {
    var pts = [], x, y;
    for (y = 0; y < bh; y++) for (x = 0; x < bw; x++) {
      if (alphaAt(b, x, y)) continue;
      if (alphaAt(b, x + 1, y) === 255 || alphaAt(b, x - 1, y) === 255 ||
          alphaAt(b, x, y + 1) === 255 || alphaAt(b, x, y - 1) === 255) pts.push([x, y]);
    }
    pts.forEach(function (p) { put(b, p[0], p[1], OUTLINE); });
  }

  function bufToCanvas(buf, w, h) {
    var c = document.createElement("canvas");
    c.width = w; c.height = h;
    var g = c.getContext("2d");
    var img = g.createImageData(w, h);
    img.data.set(buf);
    g.putImageData(img, 0, 0);
    return c;
  }

  function composeAgent(seat, phase, back) {
    bw = SW; bh = SH;
    var b = new Uint8ClampedArray(SW * SH * 4);
    var c = SEAT_COL[seat % SEAT_COL.length];
    drawTorso(b, c.shirt, back);
    drawLegs(b, phase);
    drawHead(b);
    if (!back) drawFace(b, ["flat", "sharp", "soft", "flat"][seat % 4]);
    drawHair(b, (seat % 3) + 1, c.hair);
    outline(b);
    return bufToCanvas(b, SW, SH);
  }

  var SPRITES = [];                     // [seat][phase]
  for (var si = 0; si < 4; si++) {
    SPRITES.push([0, 1, 2].map(function (p) { return composeAgent(si, p, false); }));
  }

  /* =========================================================================
     2. the scene
     ========================================================================= */

  var W = 352, H = 240, WALL = 52, SS = 3;
  cv.width = W * SS; cv.height = H * SS;

  /* Three desks across the bullpen. A run with more agents than desks reuses
     the row -- the console is a preview, not a seating chart. */
  var DESKS = [{ x: 16 }, { x: 142 }, { x: 268 }];
  var DESK_W = 68, DESK_Y = 150;
  function deskOf(i) { return { x: DESKS[i % 3].x, y: DESK_Y }; }
  // The agent stands behind the desk and is painted before it, so the desk
  // occludes them from the waist down -- they read as seated, and the desk's
  // nameplate and role banner stay clear of the sprite.
  function standOf(i)  { var d = deskOf(i); return { x: d.x + 21, y: d.y + 12 }; }
  function screenOf(i) { var d = deskOf(i); return { x: d.x + 50, y: d.y - 2 }; }
  var BOARD = { x: 104, y: 6, w: 144, h: 40 };
  function boardPort() { return { x: BOARD.x + BOARD.w / 2, y: BOARD.y + BOARD.h }; }

  /* The action filter, drawn as a gate across the floor between the desks and
     the board. Blocked tickets stop exactly on this line -- GATE_Y is derived
     from the ticket's own flight curve at GATE_T so the two can never drift
     apart. Every ticket's arc has the same from.y/mid.y/to.y, so one constant
     covers all three desks. */
  var GATE_T = 0.26, GATE_Y;
  var gateFlash = 0;

  var STARS = [];
  for (var i = 0; i < 22; i++) STARS.push([Math.random(), Math.random()]);

  function tileHash(tx, ty) { return ((tx * 31 + ty * 17) % 97 + 97) % 97; }

  function drawWindow(wx) {
    var g = ctx.createLinearGradient(0, 8, 0, 34);
    g.addColorStop(0, "#1B2340"); g.addColorStop(1, "#2E3A5C");
    ctx.fillStyle = g; ctx.fillRect(wx, 8, 44, 26);
    STARS.forEach(function (s, i) {
      if (i % 2) return;
      ctx.fillStyle = "rgba(221,225,245,0.75)";
      ctx.fillRect(wx + 2 + Math.floor(s[0] * 40), 10 + Math.floor(s[1] * 22), 1, 1);
    });
    // city lights on the skyline -- this floor only ever runs after hours
    for (var b = 0; b < 5; b++) {
      var bx = wx + 3 + b * 8, bh2 = 6 + ((b * 7) % 11);
      ctx.fillStyle = "#141A2E";
      ctx.fillRect(bx, 34 - bh2, 6, bh2);
      ctx.fillStyle = "rgba(255,214,120,0.55)";
      ctx.fillRect(bx + 1, 36 - bh2, 2, 2);
      if (b % 2) ctx.fillRect(bx + 3, 39 - bh2, 2, 2);
    }
    ctx.strokeStyle = INK; ctx.lineWidth = 2; ctx.strokeRect(wx - 1, 7, 46, 28);
    ctx.fillStyle = "rgba(26,19,32,0.55)";
    ctx.fillRect(wx + 21, 8, 2, 26); ctx.fillRect(wx, 20, 44, 2);
    ctx.fillStyle = "#3A2F52"; ctx.fillRect(wx - 3, 35, 50, 3);
  }

  /* The market board: the wall screen every order has to clear. It is the
     destination of every ticket, and it flashes when one is blocked. */
  var boardFlash = 0;
  function drawBoard(run, now) {
    ctx.fillStyle = "#0E1420";
    ctx.fillRect(BOARD.x, BOARD.y, BOARD.w, BOARD.h);
    if (boardFlash > 0.02) {
      ctx.fillStyle = "rgba(255,107,107," + (boardFlash * 0.35).toFixed(3) + ")";
      ctx.fillRect(BOARD.x, BOARD.y, BOARD.w, BOARD.h);
    }
    ctx.strokeStyle = INK; ctx.lineWidth = 2;
    ctx.strokeRect(BOARD.x, BOARD.y, BOARD.w, BOARD.h);
    ctx.fillStyle = "#232B3A";
    ctx.fillRect(BOARD.x, BOARD.y, BOARD.w, 9);

    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillStyle = GOLD;
    ctx.font = "11px VT323, monospace";
    ctx.fillText("MARKET BOARD", BOARD.x + BOARD.w / 2, BOARD.y + 5);

    // Scrolling ticker of the symbols this run's agents actually watch.
    var syms = [];
    run.console.agents.forEach(function (a) {
      a.watchlist.forEach(function (s) { if (syms.indexOf(s) < 0) syms.push(s); });
    });
    if (!syms.length) syms = ["--"];
    // Two ticker lanes scrolling opposite ways, symbol and delta on one line.
    ctx.save();
    ctx.beginPath();
    ctx.rect(BOARD.x + 2, BOARD.y + 11, BOARD.w - 4, BOARD.h - 13);
    ctx.clip();
    ctx.font = "13px VT323, monospace";
    ctx.textAlign = "left";
    var span = syms.length * 78;
    [0, 1].forEach(function (lane) {
      var scroll = lane
        ? span - ((now * 0.016 + 140) % span)
        : (now * 0.020) % span;
      for (var k = 0; k < syms.length * 2; k++) {
        var sym = syms[(k + lane) % syms.length];
        var sx = BOARD.x + 6 + k * 78 - scroll;
        if (sx > BOARD.x + BOARD.w || sx < BOARD.x - 78) continue;
        // Deterministic per symbol, so the board is stable between frames.
        var up = (sym.charCodeAt(0) + sym.length + lane) % 2 === 0;
        var y = BOARD.y + 21 + lane * 13;
        ctx.fillStyle = CREAM; ctx.fillText(sym, sx, y);
        var w = ctx.measureText(sym).width;
        ctx.fillStyle = up ? GREEN : RED;
        ctx.fillText((up ? " ▲" : " ▼") + (((sym.charCodeAt(1) % 9) + 1) / 10).toFixed(1),
                     sx + w, y);
      }
    });
    ctx.restore();
  }

  function drawWallClock(x, y, now) {
    ctx.fillStyle = "#241C2C"; ctx.beginPath(); ctx.arc(x, y, 11, 0, 6.3); ctx.fill();
    ctx.strokeStyle = INK; ctx.lineWidth = 2; ctx.stroke();
    ctx.fillStyle = "#3A2F52"; ctx.beginPath(); ctx.arc(x, y, 8, 0, 6.3); ctx.fill();
    // The hands track the replay's step, not wall time -- this clock reports
    // how far into the run the floor is.
    var frac = run.console.steps ? curStep / run.console.steps : 0;
    var ang = frac * Math.PI * 2 - Math.PI / 2;
    ctx.strokeStyle = GOLD; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(x, y);
    ctx.lineTo(x + Math.cos(ang) * 6, y + Math.sin(ang) * 6); ctx.stroke();
    ctx.strokeStyle = CREAM; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x, y);
    ctx.lineTo(x + Math.cos(ang * 12) * 4, y + Math.sin(ang * 12) * 4); ctx.stroke();
    ctx.fillStyle = CREAM; ctx.fillRect(x - 1, y - 1, 2, 2);
  }

  function drawDoor() {
    ctx.fillStyle = "#3A2B20"; ctx.fillRect(300, 10, 34, 38);
    ctx.fillStyle = "#6B4A32"; ctx.fillRect(303, 13, 28, 33);
    ctx.fillStyle = "#7D583C"; ctx.fillRect(306, 16, 22, 11);
    ctx.fillStyle = "#4B3524"; ctx.fillRect(306, 30, 22, 11);
    ctx.fillStyle = GOLD; ctx.fillRect(326, 26, 3, 3);
    ctx.strokeStyle = INK; ctx.lineWidth = 2; ctx.strokeRect(300, 10, 34, 38);
    ctx.fillStyle = INK; ctx.fillRect(308, 2, 20, 8);
    ctx.font = "10px VT323, monospace"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillStyle = GREEN; ctx.fillText("EXIT", 318, 6);
  }

  function drawPlant(x, y) {
    ctx.fillStyle = "#8A5730"; ctx.fillRect(x, y - 8, 12, 8);
    ctx.strokeStyle = INK; ctx.lineWidth = 2; ctx.strokeRect(x, y - 8, 12, 8);
    ctx.fillStyle = "#2F9E6E";
    ctx.fillRect(x + 4, y - 21, 4, 13);
    ctx.fillRect(x, y - 16, 4, 8);
    ctx.fillRect(x + 8, y - 15, 4, 7);
    ctx.fillStyle = GREEN;
    ctx.fillRect(x + 5, y - 21, 2, 4); ctx.fillRect(x + 1, y - 16, 2, 3);
  }

  function drawDesk(i, run, now) {
    var d = deskOf(i), a = run.console.agents[i];
    ctx.fillStyle = "#A86F3F"; ctx.fillRect(d.x, d.y, DESK_W, 16);
    ctx.fillStyle = "#B9804E"; ctx.fillRect(d.x, d.y, DESK_W, 3);
    ctx.fillStyle = "#8A5730"; ctx.fillRect(d.x, d.y + 16, DESK_W, 9);
    ctx.strokeStyle = INK; ctx.lineWidth = 2; ctx.strokeRect(d.x, d.y, DESK_W, 25);

    // monitor, facing the agent -- we see the back and the light spilling round
    var sc = screenOf(i);
    var flick = 0.30 + 0.18 * (0.5 + 0.5 * Math.sin(now / 340 + i));
    var lit = flashUntil[i] && now < flashUntil[i];
    ctx.fillStyle = lit ? "rgba(255,107,107,0.95)"
                        : "rgba(170,235,210," + flick.toFixed(2) + ")";
    ctx.fillRect(sc.x - 13, sc.y - 12, 26, 2);
    ctx.fillRect(sc.x - 14, sc.y - 10, 2, 12);
    ctx.fillRect(sc.x + 12, sc.y - 10, 2, 12);
    ctx.fillStyle = "#2C2C34"; ctx.fillRect(sc.x - 12, sc.y - 10, 24, 13);
    ctx.fillStyle = "#3A3A46"; ctx.fillRect(sc.x - 12, sc.y - 10, 24, 2);
    ctx.strokeStyle = INK; ctx.lineWidth = 2; ctx.strokeRect(sc.x - 12, sc.y - 10, 24, 13);
    ctx.fillStyle = "#1C1C26"; ctx.fillRect(sc.x - 4, sc.y + 3, 8, 3);
    ctx.fillStyle = "#2C2C34"; ctx.fillRect(sc.x - 8, sc.y + 6, 16, 2);

    // mug and a sticky note, because a desk with nothing on it reads as a prop
    ctx.fillStyle = RED;  ctx.fillRect(d.x + 58, d.y + 5, 5, 6);
    ctx.fillStyle = TEAL; ctx.fillRect(d.x + 5, d.y + 6, 5, 5);

    // nameplate + a LED that is amber while this agent has a ticket in flight
    ctx.fillStyle = CREAM; ctx.fillRect(d.x + 14, d.y + 15, 40, 11);
    ctx.strokeStyle = INK; ctx.lineWidth = 1.5; ctx.strokeRect(d.x + 14.5, d.y + 15.5, 39, 10);
    ctx.font = "11px VT323, monospace"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillStyle = INK;
    ctx.fillText(a.name.split(" ")[0].toUpperCase(), d.x + 34, d.y + 21);
    ctx.fillStyle = busy[i] ? GOLD : "#2F9E6E";
    ctx.fillRect(d.x + 9, d.y + 19, 4, 4);

    // role banner, in the seat colour
    ctx.font = "10px VT323, monospace";
    var tag = (a.role || "trader").toUpperCase();
    var tw = ctx.measureText(tag).width + 10;
    var bx = Math.round(d.x + 34 - tw / 2);
    ctx.fillStyle = INK;  ctx.fillRect(bx + 2, d.y + 29, tw, 11);
    ctx.fillStyle = SEAT_COL[i % SEAT_COL.length].tag;
    ctx.fillRect(bx, d.y + 27, tw, 11);
    ctx.strokeStyle = INK; ctx.lineWidth = 1.5; ctx.strokeRect(bx + 0.5, d.y + 27.5, tw - 1, 10);
    ctx.fillStyle = INK; ctx.fillText(tag, bx + tw / 2, d.y + 33);
  }

  function drawAgent(i, now) {
    var p = standOf(i);
    // A one-pixel bob at a per-seat offset; without the offset the three of
    // them breathe in lockstep and the floor looks mechanical.
    var phase = busy[i] ? (Math.floor(now / 160) % 3) : 0;
    var bob = Math.floor((now + i * 220) / 500) % 2;
    ctx.drawImage(SPRITES[i % 4][phase], Math.round(p.x) - 8, Math.round(p.y) - SH + bob);
  }

  /* Drawn after the desks so it is never occluded by the furniture. */
  function drawChevron(i, now) {
    if (i !== sel) return;
    var p = standOf(i);
    var bob = Math.floor(now / 300) % 2;
    ctx.fillStyle = GOLD;
    ctx.beginPath();
    ctx.moveTo(p.x - 5, p.y - SH - 11 - bob);
    ctx.lineTo(p.x + 5, p.y - SH - 11 - bob);
    ctx.lineTo(p.x, p.y - SH - 4 - bob);
    ctx.closePath(); ctx.fill();
    ctx.strokeStyle = INK; ctx.lineWidth = 1.5; ctx.stroke();
  }

  function drawChair(i) {
    var p = standOf(i);
    ctx.fillStyle = "#4A3A30"; ctx.fillRect(p.x - 9, p.y - 30, 18, 5);
    ctx.strokeStyle = INK; ctx.lineWidth = 1.5;
    ctx.strokeRect(p.x - 8.5, p.y - 29.5, 17, 4);
  }

  /* =========================================================================
     3. replay
     ========================================================================= */

  var run = RUNS[0], sel = 0;
  var tickets = [], toasts = [], busy = [], flashUntil = [];
  var cursor = 0, nextAt = 0, curStep = 0;
  var STEP_MS = 900;

  function resetReplay() {
    tickets.length = 0; toasts.length = 0;
    busy = []; flashUntil = [];
    cursor = 0; nextAt = 0; curStep = 0;
    boardFlash = 0;
  }

  function qbez(a, c, b, t) {
    var u = 1 - t;
    return { x: u * u * a.x + 2 * u * t * c.x + t * t * b.x,
             y: u * u * a.y + 2 * u * t * c.y + t * t * b.y };
  }

  (function () {
    var from = { x: 0, y: DESK_Y - 2 }, to = boardPort();
    var mid = { x: 0, y: Math.min(from.y, to.y) - 30 };
    GATE_Y = qbez(from, mid, to, GATE_T).y;
  })();

  function drawGate(now) {
    var y = Math.round(GATE_Y);
    ctx.save();
    ctx.setLineDash([6, 5]);
    ctx.lineWidth = 2;
    ctx.strokeStyle = gateFlash > 0.02
      ? "rgba(255,107,107," + (0.45 + gateFlash * 0.55).toFixed(2) + ")"
      : "rgba(255,202,84,0.42)";
    ctx.beginPath(); ctx.moveTo(14, y + 0.5); ctx.lineTo(W - 14, y + 0.5); ctx.stroke();
    ctx.restore();

    ctx.font = "11px VT323, monospace";
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    var label = "ACTION FILTER";
    var tw = ctx.measureText(label).width + 12;
    var lx = 62;
    ctx.fillStyle = "#241C2C";
    ctx.fillRect(lx - tw / 2, y - 7, tw, 14);
    ctx.strokeStyle = gateFlash > 0.02 ? RED : "rgba(255,202,84,0.5)";
    ctx.lineWidth = 1;
    ctx.strokeRect(lx - tw / 2 + 0.5, y - 6.5, tw - 1, 13);
    ctx.fillStyle = gateFlash > 0.02 ? RED : "rgba(255,202,84,0.85)";
    ctx.fillText(label, lx, y);
  }

  function fire(rec, now) {
    var i = rec.a % 3;
    curStep = rec.step;
    if (rec.action === "hold" && !rec.halluc) {
      // A hold is a decision too, and a floor where two thirds of the steps
      // show nothing would misrepresent the run. It gets a quiet bubble.
      toasts.push({ x: standOf(i).x, y: standOf(i).y - SH - 16, text: "HOLD",
                    t: 0, colour: "#717A8C" });
      return;
    }
    busy[i] = true;
    var label = (rec.req || rec.action).toUpperCase() +
                (rec.symbol ? " " + rec.symbol : "") +
                (rec.qty ? " x" + rec.qty : "");
    tickets.push({
      from: screenOf(i), to: boardPort(), t: 0, seat: i,
      blocked: rec.halluc, kind: rec.kind, label: label, stamped: false
    });
  }

  function stepReplay(now) {
    var tl = run.console.timeline;
    if (!tl.length) return;
    if (now >= nextAt) {
      fire(tl[cursor], now);
      cursor = (cursor + 1) % tl.length;
      if (cursor === 0) resetReplay();          // loop the run
      nextAt = now + STEP_MS;
    }
    for (var i = tickets.length - 1; i >= 0; i--) {
      var tk = tickets[i];
      tk.t += 1 / 46;
      // A blocked ticket only travels 62% of the way -- it never reaches the
      // board, which is the entire point of the middleware.
      var limit = tk.blocked ? GATE_T : 1;
      if (tk.t >= limit && !tk.stamped) {
        tk.stamped = true;
        busy[tk.seat] = false;
        if (tk.blocked) {
          gateFlash = 1;
          flashUntil[tk.seat] = now + 500;
          toasts.push({ x: standOf(tk.seat).x, y: standOf(tk.seat).y - SH - 18,
                        text: "BLOCKED" + (tk.kind ? " · " + tk.kind.replace(/_/g, " ") : ""),
                        t: 0, colour: RED });
        } else {
          boardFlash = 1;
          toasts.push({ x: boardPort().x, y: BOARD.y + BOARD.h + 24,
                        text: "✓ FILLED", t: 0, colour: "#2F9E6E" });
        }
      }
      if (tk.t >= limit + 0.45) tickets.splice(i, 1);
    }
    boardFlash *= 0.90;
    gateFlash *= 0.90;
    for (i = toasts.length - 1; i >= 0; i--) {
      toasts[i].t += 16;
      if (toasts[i].t > 1900) toasts.splice(i, 1);
    }
  }

  function drawTickets() {
    tickets.forEach(function (tk) {
      var limit = tk.blocked ? GATE_T : 1;
      var t = Math.min(tk.t, limit);
      var mid = { x: (tk.from.x + tk.to.x) / 2, y: Math.min(tk.from.y, tk.to.y) - 30 };
      // trail
      for (var g = 3; g >= 1; g--) {
        var gt = t - g * 0.05;
        if (gt <= 0) continue;
        var gp = qbez(tk.from, mid, tk.to, gt);
        ctx.fillStyle = (tk.blocked ? "rgba(255,107,107," : "rgba(107,207,127,") +
                        (0.45 - g * 0.12) + ")";
        ctx.fillRect(Math.round(gp.x) - 1, Math.round(gp.y) - 1, 3, 3);
      }
      var p = qbez(tk.from, mid, tk.to, t);
      // Once stamped, a blocked ticket sags a little and fades -- enough to
      // read as rejected without drifting off the filter line it died on.
      var over = tk.stamped ? tk.t - limit : 0;
      var fall = tk.blocked ? over * 26 : 0;
      var alpha = tk.stamped ? Math.max(0, 1 - over / 0.45) : 1;

      // The ticket is sized to its own text, so "SELL TSLA x1" is legible
      // rather than clipped to a fixed box.
      ctx.font = "11px VT323, monospace";
      var tw = Math.max(ctx.measureText(tk.label).width + 10, 34);
      ctx.save();
      ctx.globalAlpha = alpha;
      var x = Math.round(p.x - tw / 2), y = Math.round(p.y) - 8 + fall;
      ctx.fillStyle = "rgba(26,19,32,0.5)"; ctx.fillRect(x + 2, y + 2, tw, 16);
      ctx.fillStyle = CREAM; ctx.fillRect(x, y, tw, 16);
      ctx.fillStyle = tk.blocked ? RED : "#2F9E6E";
      ctx.fillRect(x, y, tw, 4);
      ctx.strokeStyle = INK; ctx.lineWidth = 1.5;
      ctx.strokeRect(x + 0.5, y + 0.5, tw - 1, 15);
      ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillStyle = INK;
      ctx.fillText(tk.label, x + tw / 2, y + 10.5);
      if (tk.stamped && tk.blocked) {                    // the rejection stamp
        ctx.strokeStyle = RED; ctx.lineWidth = 2;
        ctx.strokeRect(x - 2.5, y - 2.5, tw + 5, 21);
        ctx.beginPath();
        ctx.moveTo(x - 2, y - 2); ctx.lineTo(x + tw + 2, y + 18);
        ctx.stroke();
      }
      ctx.restore();
    });
  }

  function drawToasts() {
    for (var i = toasts.length - 1; i >= 0; i--) {
      var to = toasts[i];
      var rise = to.t * 0.008;
      var al = to.t < 200 ? to.t / 200 : to.t > 1500 ? Math.max(0, (1900 - to.t) / 400) : 1;
      ctx.font = "11px VT323, monospace";
      var tw = ctx.measureText(to.text).width + 10;
      var x = Math.round(to.x - tw / 2), y = Math.round(to.y - rise);
      ctx.globalAlpha = al;
      ctx.fillStyle = INK;   ctx.fillRect(x + 2, y + 2, tw, 14);
      ctx.fillStyle = CREAM; ctx.fillRect(x, y, tw, 14);
      ctx.strokeStyle = INK; ctx.lineWidth = 1.5; ctx.strokeRect(x + 0.5, y + 0.5, tw - 1, 13);
      ctx.fillStyle = to.colour;
      ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText(to.text, x + tw / 2, y + 7.5);
      ctx.globalAlpha = 1;
    }
  }

  /* ---- the frame -------------------------------------------------------- */

  function drawScene(now) {
    ctx.setTransform(SS, 0, 0, SS, 0, 0);
    ctx.imageSmoothingEnabled = false;

    // wall
    ctx.fillStyle = "#2E2540"; ctx.fillRect(0, 0, W, WALL);
    ctx.fillStyle = "#352B4A"; ctx.fillRect(0, 0, W, 4);
    ctx.fillStyle = "#241C2C"; ctx.fillRect(0, WALL - 9, W, 9);
    ctx.fillStyle = "#3A2F52"; ctx.fillRect(0, WALL - 2, W, 2);
    drawWindow(14);
    drawWallClock(268, 26, now);
    drawBoard(run, now);
    drawDoor();

    // floor: two-tone tiles with a deterministic speckle so it is not flat
    for (var ty = WALL; ty < H; ty += 16) for (var tx = 0; tx < W; tx += 16) {
      ctx.fillStyle = ((tx + ty) / 16) % 2 ? "#3A3050" : "#342B48";
      ctx.fillRect(tx, ty, 16, 16);
      var h = tileHash(tx / 16, ty / 16);
      if (h < 14) { ctx.fillStyle = "rgba(26,19,32,0.06)"; ctx.fillRect(tx, ty, 16, 16); }
      else if (h > 88) { ctx.fillStyle = "rgba(255,248,231,0.04)"; ctx.fillRect(tx, ty, 16, 16); }
      if (h % 11 === 0) { ctx.fillStyle = "rgba(26,19,32,0.14)"; ctx.fillRect(tx + (h % 13), ty + (h % 9), 2, 1); }
    }
    ctx.fillStyle = "rgba(26,19,32,0.22)"; ctx.fillRect(0, WALL, W, 3);

    // rug under the desk row
    ctx.fillStyle = "#463A5E"; ctx.fillRect(8, 138, 336, 92);
    ctx.strokeStyle = "#5A4B78"; ctx.lineWidth = 3; ctx.strokeRect(9.5, 139.5, 333, 89);
    ctx.strokeStyle = "rgba(26,19,32,0.25)"; ctx.lineWidth = 1; ctx.strokeRect(14.5, 144.5, 323, 79);

    // board light washing down the wall onto the floor
    ctx.fillStyle = "rgba(255,214,120,0.05)";
    ctx.beginPath();
    ctx.moveTo(BOARD.x + 6, WALL); ctx.lineTo(BOARD.x + BOARD.w - 6, WALL);
    ctx.lineTo(BOARD.x + BOARD.w + 22, 138); ctx.lineTo(BOARD.x - 22, 138);
    ctx.closePath(); ctx.fill();

    drawPlant(8, 132);
    drawPlant(332, 132);

    // Agents first, desks over them: that occlusion is what seats them.
    var n = Math.min(run.console.agents.length, 3);
    for (var i = 0; i < n; i++) drawChair(i);
    for (i = 0; i < n; i++) drawAgent(i, now);
    for (i = 0; i < n; i++) drawDesk(i, run, now);
    for (i = 0; i < n; i++) drawChevron(i, now);

    drawGate(now);
    drawTickets();
    drawToasts();

    // Step readout, bottom-right -- the bottom-left corner belongs to the
    // CLICK AN AGENT hint, which is a DOM element sitting over the canvas.
    ctx.font = "13px VT323, monospace";
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    var txt = "STEP " + curStep + " / " + Math.max(0, run.console.steps - 1);
    var tw = ctx.measureText(txt).width;
    ctx.fillStyle = "rgba(26,19,32,0.72)";
    ctx.fillRect(W - tw - 16, H - 21, tw + 12, 15);
    ctx.strokeStyle = "rgba(255,248,231,0.18)"; ctx.lineWidth = 1;
    ctx.strokeRect(W - tw - 15.5, H - 20.5, tw + 11, 14);
    ctx.fillStyle = CREAM;
    ctx.fillText(txt, W - 10, H - 13);
  }

  /* ---- card + controls -------------------------------------------------- */

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }
  function set(id, html, cls) {
    var el = document.getElementById(id);
    if (!el) return;
    el.innerHTML = html;
    if (cls !== undefined) el.className = cls;
  }

  var tabs  = document.getElementById("idTabs");
  var btns  = document.getElementById("stBtns");
  var read  = document.getElementById("dexRead");
  var title = document.getElementById("dexTitle");

  function renderCard() {
    var a = run.console.agents[sel];
    if (!a) return;
    document.getElementById("idSerial").innerHTML = "DESK&nbsp;#" + ("0" + (sel + 1)).slice(-2);
    document.getElementById("idName").textContent = a.name;
    document.getElementById("idRole").textContent =
      a.role + (a.risk ? " · " + a.risk + " risk" : "");
    set("idRun", esc(run.sim_code));
    set("idWatch", a.watchlist.length ? esc(a.watchlist.join(" ")) : "—");
    set("idTrades", String(a.trades));
    set("idHolds", String(a.holds));
    set("idFlag", a.flagged ? "● " + a.flagged + " infeasible" : "● none",
        a.flagged ? "flag" : "on");
    set("idLast", a.last ? esc(a.last) : "—");

    tabs.innerHTML = "";
    run.console.agents.forEach(function (ag, i) {
      var b = document.createElement("button");
      b.textContent = ag.name.split(" ")[0];
      b.className = i === sel ? "on" : "";
      b.onclick = function () { sel = i; renderCard(); };
      tabs.appendChild(b);
    });

    read.textContent = run.console.agents.length + " AGENTS · " +
                       run.decisions + " DECISIONS · " + run.console.flagged + " FLAGGED";
    title.textContent = "ANCHOR-1 · " + run.sim_code.toUpperCase();
  }

  RUNS.slice(0, 3).forEach(function (r, i) {
    var b = document.createElement("button");
    b.className = "st-btn" + (i === 0 ? " on" : "");
    b.textContent = r.sim_code.toUpperCase();
    b.onclick = function () {
      run = r; sel = 0;
      resetReplay();
      [].forEach.call(btns.children, function (c) { c.classList.remove("on"); });
      b.classList.add("on");
      renderCard();
      if (reduced) drawScene(1000);
    };
    btns.appendChild(b);
  });

  // Clicking an agent on the floor selects them, same as the tabs.
  cv.addEventListener("click", function (e) {
    var r = cv.getBoundingClientRect();
    var x = (e.clientX - r.left) * (W / r.width);
    var y = (e.clientY - r.top) * (H / r.height);
    for (var i = 0; i < Math.min(run.console.agents.length, 3); i++) {
      var p = standOf(i);
      if (x >= p.x - 12 && x <= p.x + 12 && y >= p.y - SH - 6 && y <= p.y + 4) {
        sel = i; renderCard(); return;
      }
    }
  });

  /* ---- loop ------------------------------------------------------------- */

  /* Fast-forward the replay to a given virtual time and paint one frame.
     The replay is a pure function of elapsed ms, so a seek reproduces exactly
     what the live loop would show at that moment. Used for the reduced-motion
     still, and by #t=<ms> to freeze the console on a particular beat -- which
     is how you park it on a BLOCKED ticket for a demo. */
  function seekTo(ms) {
    resetReplay();
    for (var t = 0; t <= ms; t += 16) stepReplay(t);
    drawScene(ms);
  }

  function hashSeek() {
    var m = /(?:^|[#&])t=(\d+)/.exec(location.hash);
    return m ? parseInt(m[1], 10) : null;
  }

  var running = true;
  function frame(ts) {
    if (!running) return;
    stepReplay(ts);
    drawScene(ts);
    requestAnimationFrame(frame);
  }

  renderCard();
  var frozen = hashSeek();
  if (frozen !== null) {
    seekTo(frozen);
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(function () { seekTo(frozen); });
    }
    window.addEventListener("hashchange", function () {
      var f = hashSeek();
      if (f !== null) seekTo(f);
    });
  } else if (reduced) {
    // Draw the room once, with no motion, and leave it there.
    seekTo(1000);
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(function () { seekTo(1000); });
    }
  } else {
    requestAnimationFrame(frame);
    document.addEventListener("visibilitychange", function () {
      var was = running;
      running = !document.hidden;
      if (running && !was) requestAnimationFrame(frame);
    });
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(function () { drawScene(performance.now()); });
    }
  }
})();
