/*
  Procedural character sprites for the trading floor's three agents (Alex Chen,
  Marcus Webb, Sara Kim), matching the look of Munder Difflin's office cast --
  fully custom-drawn pixel-art busts (skin -> clothing -> face -> hair layered
  on an 18x32 canvas), NOT LimeZu sprites. Ported from that project's
  scene/office/portraitArt.ts (original: canvas ImageData -> Pixi Texture; here:
  canvas ImageData -> a Phaser texture with the SAME frame-name contract this
  file's atlas.json already used, so main_script.html's animation/facing code
  needs no changes -- see build_cast_texture() at the bottom).

  This is original code (not a LimeZu recolor) -- see ATTRIBUTION.md in
  static_dirs/assets/office/visuals/ for the map/tileset's separate, restricted
  license; this file carries no such restriction.
*/
(function (global) {
  "use strict";

  var SCENE_W = 18, SCENE_H = 32;
  var HX0 = 4, HX1 = 13;
  var OUTLINE = [38, 34, 46];

  function clamp(v) { return v < 0 ? 0 : v > 255 ? 255 : Math.round(v); }
  function shades(rgb, dl, dd) {
    dl = dl == null ? 1.22 : dl;
    dd = dd == null ? 0.68 : dd;
    return [
      [clamp(rgb[0] * dl), clamp(rgb[1] * dl), clamp(rgb[2] * dl)],
      [rgb[0], rgb[1], rgb[2]],
      [clamp(rgb[0] * dd), clamp(rgb[1] * dd), clamp(rgb[2] * dd)]
    ];
  }
  function setpx(buf, x, y, c, a) {
    if (x < 0 || x >= SCENE_W || y < 0 || y >= SCENE_H) return;
    var i = (y * SCENE_W + x) * 4;
    buf[i] = c[0]; buf[i + 1] = c[1]; buf[i + 2] = c[2]; buf[i + 3] = (a == null ? 255 : a);
  }
  function alphaAt(buf, x, y) {
    if (x < 0 || x >= SCENE_W || y < 0 || y >= SCENE_H) return 0;
    return buf[(y * SCENE_W + x) * 4 + 3];
  }
  function rect(buf, x0, y0, x1, y1, c) {
    for (var y = y0; y <= y1; y++) for (var x = x0; x <= x1; x++) setpx(buf, x, y, c);
  }

  var SKIN = {
    light: { hi: [255, 221, 189], base: [247, 201, 170], sh: [212, 158, 126], line: [168, 112, 82] },
    tan:   { hi: [232, 182, 136], base: [214, 162, 116], sh: [176, 126, 86],  line: [138, 92, 60] },
    brown: { hi: [180, 130, 94],  base: [158, 112, 78],  sh: [124, 86, 58],   line: [90, 60, 40] },
    dark:  { hi: [142, 98, 70],   base: [120, 80, 56],   sh: [94, 62, 42],    line: [64, 42, 28] }
  };

  // ─── head + face (front view) ──────────────────────────────────────────────
  function drawHead(buf, skin) {
    var s = SKIN[skin];
    for (var y = 4; y <= 16; y++) {
      for (var x = HX0; x <= HX1; x++) {
        if (((x === HX0 || x === HX1) && (y === 4 || y === 5 || y === 16)) || ((x === 5 || x === 12) && y === 4)) continue;
        setpx(buf, x, y, s.base);
      }
    }
    for (var y2 = 6; y2 < 12; y2++) setpx(buf, 5, y2, s.hi);
    setpx(buf, 6, 5, s.hi); setpx(buf, 7, 5, s.hi);
    for (var y3 = 6; y3 < 15; y3++) setpx(buf, 12, y3, s.sh);
    [7, 8, 9, 10, 11].forEach(function (x) { setpx(buf, x, 16, s.sh); });
    [HX0 - 1, HX1 + 1].forEach(function (ex) {
      setpx(buf, ex, 9, s.base); setpx(buf, ex, 10, s.base); setpx(buf, ex, 11, s.sh);
    });
    rect(buf, 7, 17, 10, 18, s.sh); rect(buf, 7, 17, 9, 17, s.base);
  }

  function drawFace(buf, skin, brow, mouth, blush, lashes) {
    var s = SKIN[skin];
    var white = [250, 248, 244], pup = [46, 38, 42];
    [[5, 6, 6], [10, 11, 10]].forEach(function (t) {
      setpx(buf, t[0], 9, white); setpx(buf, t[1], 9, white); setpx(buf, t[2], 9, pup);
    });
    if (lashes) {
      var lash = [54, 40, 48], glint = [252, 250, 248];
      [5, 6, 10, 11].forEach(function (x) { setpx(buf, x, 8, lash); });
      setpx(buf, 4, 8, lash); setpx(buf, 12, 8, lash);
      setpx(buf, 5, 9, glint); setpx(buf, 10, 9, glint);
    }
    if (brow === "flat") [5, 6, 10, 11].forEach(function (x) { setpx(buf, x, 7, s.line); });
    else if (brow === "angry") { setpx(buf, 5, 8, s.line); setpx(buf, 6, 7, s.line); setpx(buf, 10, 7, s.line); setpx(buf, 11, 8, s.line); }
    else if (brow === "raised") [5, 6, 10, 11].forEach(function (x) { setpx(buf, x, 6, s.line); });
    else if (brow === "soft") { [5, 11].forEach(function (x) { setpx(buf, x, 7, s.line); }); [6, 10].forEach(function (x) { setpx(buf, x, 7, s.sh); }); }
    setpx(buf, 8, 11, s.sh); setpx(buf, 8, 12, s.sh); setpx(buf, 7, 12, s.sh);
    var mc = [158, 86, 80];
    var mouths = {
      neutral: [[7, 14], [8, 14], [9, 14], [10, 14]],
      smile: [[7, 14], [8, 14], [9, 14], [10, 14], [6, 13], [11, 13]],
      frown: [[7, 15], [8, 15], [9, 15], [10, 15], [6, 14], [11, 14]],
      grin: [[7, 14], [8, 14], [9, 14], [10, 14], [7, 13], [8, 13], [9, 13], [10, 13], [6, 13], [11, 13]]
    };
    mouths[mouth].forEach(function (p) { setpx(buf, p[0], p[1], mc); });
    if (blush) [5, 12].forEach(function (x) { setpx(buf, x, 12, [235, 150, 140], 140); });
  }

  // ─── hairstyles (front view; back view handled separately by drawHeadBack) ──
  var HAIR = {
    styleShort: function (buf, color, skinBase, a) {
      var sh3 = shades(color), hi = sh3[0], base = sh3[1], sh = sh3[2];
      var part = a.part || "L";
      rect(buf, HX0, 2, HX1, 4, base);
      for (var x = HX0 - 1; x <= HX1 + 1; x++) setpx(buf, x, 3, base);
      rect(buf, HX0 - 1, 4, HX1 + 1, 5, base);
      for (var y = 6; y < 9; y++) { setpx(buf, HX0 - 1, y, base); setpx(buf, HX0, y, base); setpx(buf, HX1, y, base); setpx(buf, HX1 + 1, y, base); }
      for (var x2 = HX0; x2 <= HX1; x2++) setpx(buf, x2, 5, base);
      var hx = part === "L" ? 6 : 11;
      for (var y2 = 2; y2 < 6; y2++) setpx(buf, hx, y2, sh);
      for (var x3 = HX0; x3 < hx; x3++) if (alphaAt(buf, x3, 3)) setpx(buf, x3, 3, hi);
      for (var x4 = HX0; x4 <= HX1; x4++) if (alphaAt(buf, x4, 2)) setpx(buf, x4, 2, hi);
    },
    styleMessy: function (buf, color, skinBase, a) {
      var sh3 = shades(color), hi = sh3[0], base = sh3[1];
      var length = a.length || 8;
      rect(buf, HX0 - 1, 2, HX1 + 1, 5, base);
      var spikes = [[3, 2], [5, 1], [7, 2], [9, 1], [11, 2], [13, 1], [14, 2], [4, 2], [12, 2]];
      spikes.forEach(function (p) { setpx(buf, p[0], p[1], base); });
      for (var x = HX0; x <= HX1; x++) setpx(buf, x, 5, base);
      for (var x2 = 6; x2 < 12; x2++) setpx(buf, x2, 6, base);
      setpx(buf, 8, 6, skinBase); setpx(buf, 9, 6, skinBase);
      for (var y = 6; y <= length; y++) { setpx(buf, HX0 - 1, y, base); setpx(buf, HX0, y, base); setpx(buf, HX1, y, base); setpx(buf, HX1 + 1, y, base); }
      spikes.forEach(function (p) { setpx(buf, p[0], p[1], hi); });
    },
    styleFrame: function (buf, color, skinBase, a) {
      var sh3 = shades(color), hi = sh3[0], base = sh3[1], sh = sh3[2];
      var length = a.length || 17, vol = a.vol || 1;
      rect(buf, HX0 - 1, 2, HX1 + 1, 5, base);
      for (var x = HX0 - 1; x <= HX1 + 1; x++) setpx(buf, x, 3, base);
      for (var x2 = HX0; x2 <= HX1; x2++) setpx(buf, x2, 5, base);
      for (var x3 = 6; x3 < 12; x3++) setpx(buf, x3, 6, base);
      setpx(buf, 8, 6, skinBase); setpx(buf, 9, 6, skinBase);
      for (var y = 6; y <= length; y++) {
        for (var dx = 0; dx < vol; dx++) { setpx(buf, HX0 - 1 - dx, y, base); setpx(buf, HX1 + 1 + dx, y, base); }
        setpx(buf, HX0, y, base); setpx(buf, HX1, y, base);
      }
      for (var x4 = HX0 - 1; x4 < HX0 + 1; x4++) setpx(buf, x4, length + 1, base);
      for (var x5 = HX1; x5 < HX1 + 2; x5++) setpx(buf, x5, length + 1, base);
      for (var y2 = 2; y2 < 6; y2++) if (alphaAt(buf, HX1, y2)) setpx(buf, HX1, y2, sh);
      for (var x6 = HX0; x6 < 9; x6++) if (alphaAt(buf, x6, 2)) setpx(buf, x6, 2, hi);
    }
  };

  // ─── clothing ────────────────────────────────────────────────────────────
  function bodyShape(buf, col, heavy) {
    var sh3 = shades(col), base = sh3[1], sh = sh3[2];
    var rows = heavy
      ? [[19, 5, 12], [20, 3, 14], [21, 2, 15], [22, 1, 16], [23, 1, 16], [24, 0, 17], [25, 0, 17], [26, 0, 17], [27, 0, 17]]
      : [[19, 6, 11], [20, 4, 13], [21, 3, 14], [22, 2, 15], [23, 2, 15], [24, 1, 16], [25, 1, 16], [26, 1, 16], [27, 1, 16]];
    rows.forEach(function (r) { rect(buf, r[1], r[0], r[2], r[0], base); });
  }
  function defaultPants(r) {
    if (r.pants) return r.pants;
    return r.cloth === "suit" ? shades(r.c1)[2] : [54, 56, 70];
  }

  // ─── scene body: torso (front/back) + legs ─────────────────────────────────
  var SHOE = [44, 40, 48];
  function drawSceneLegs(buf, pants, phase) {
    var sh3 = shades(pants), base = sh3[1], sh = sh3[2];
    [[5, 7], [10, 12]].forEach(function (lr) {
      rect(buf, lr[0], 25, lr[1], 30, base);
      for (var y = 25; y <= 30; y++) setpx(buf, lr[1], y, sh);
    });
    var leftLow = phase !== 1, rightLow = phase !== 2;
    rect(buf, 5, leftLow ? 31 : 30, 7, leftLow ? 31 : 30, SHOE);
    rect(buf, 10, rightLow ? 31 : 30, 12, rightLow ? 31 : 30, SHOE);
  }
  function drawSceneTorso(buf, r, back) {
    var sh3 = shades(r.c1), hi = sh3[0], base = sh3[1], sh = sh3[2];
    if (r.heavy) {
      rect(buf, 3, 18, 14, 18, base);
      rect(buf, 2, 19, 15, 19, base);
      rect(buf, 2, 20, 15, 24, base);
      for (var y = 20; y <= 24; y++) { setpx(buf, 2, y, sh); setpx(buf, 15, y, sh); setpx(buf, 14, y, sh); }
    } else {
      rect(buf, 4, 18, 13, 18, base);
      rect(buf, 3, 19, 14, 19, base);
      rect(buf, 4, 20, 13, 24, base);
      for (var y2 = 20; y2 <= 24; y2++) { setpx(buf, 3, y2, sh); setpx(buf, 14, y2, sh); setpx(buf, 13, y2, sh); }
    }
    if (back) {
      rect(buf, 6, 18, 11, 18, sh);
      for (var y3 = 19; y3 <= 24; y3++) setpx(buf, 8, y3, sh);
      return;
    }
    var skin = SKIN[r.skin];
    if (r.cloth === "suit") {
      var white = [238, 238, 236];
      [[8, 18], [9, 18], [7, 19], [8, 19], [9, 19], [10, 19], [8, 20], [9, 20]].forEach(function (p) { setpx(buf, p[0], p[1], white); });
      [[6, 19], [7, 20], [11, 19], [10, 20]].forEach(function (p) { setpx(buf, p[0], p[1], sh); });
      if (r.tie) { for (var y4 = 19; y4 <= 24; y4++) { setpx(buf, 8, y4, r.tie); setpx(buf, 9, y4, r.tie); } setpx(buf, 8, 19, shades(r.tie)[0]); }
    } else if (r.cloth === "dressshirt") {
      [[6, 18], [7, 18], [10, 18], [11, 18], [7, 19], [10, 19]].forEach(function (p) { setpx(buf, p[0], p[1], sh); });
      if (r.tie) for (var y5 = 18; y5 <= 24; y5++) { setpx(buf, 8, y5, r.tie); setpx(buf, 9, y5, r.tie); }
      else for (var y6 = 20; y6 <= 24; y6 += 2) setpx(buf, 8, y6, sh);
    } else if (r.cloth === "polo") {
      [[6, 18], [7, 18], [10, 18], [11, 18]].forEach(function (p) { setpx(buf, p[0], p[1], hi); });
      setpx(buf, 8, 19, sh); setpx(buf, 8, 21, sh);
    } else if (r.cloth === "blouse") {
      [[7, 18], [8, 18], [9, 18], [10, 18], [8, 19], [9, 19]].forEach(function (p) { setpx(buf, p[0], p[1], skin.sh); });
      for (var x = 5; x < 13; x++) {
        var i = ((19) * SCENE_W + x) * 4;
        if (buf[i] === base[0] && buf[i + 1] === base[1] && buf[i + 2] === base[2]) setpx(buf, x, 19, hi);
      }
    } else if (r.cloth === "cardigan") {
      var inner = r.c2 ? shades(r.c2)[1] : [235, 233, 226];
      for (var y7 = 18; y7 <= 24; y7++) { setpx(buf, 8, y7, inner); setpx(buf, 9, y7, inner); }
      [[6, 18], [7, 18], [10, 18], [11, 18]].forEach(function (p) { setpx(buf, p[0], p[1], sh); });
    } else if (r.cloth === "sweater") {
      [[6, 18], [7, 18], [8, 18], [9, 18], [10, 18], [11, 18]].forEach(function (p) { setpx(buf, p[0], p[1], sh); });
    }
  }

  /** Back of the head: rounded hair-covered skull, no face. */
  function drawHeadBack(buf, r) {
    var s = SKIN[r.skin];
    var sh3 = shades(r.hairc), hi = sh3[0], base = sh3[1], sh = sh3[2];
    var rows = [
      [2, 6, 11], [3, 5, 12], [4, 4, 13], [5, 4, 13], [6, 4, 13], [7, 4, 13], [8, 4, 13],
      [9, 4, 13], [10, 4, 13], [11, 4, 13], [12, 4, 13], [13, 5, 12], [14, 6, 11]
    ];
    rows.forEach(function (row) { rect(buf, row[1], row[0], row[2], row[0], base); });
    var len = r.hair === "styleFrame" ? ((r.hairargs && r.hairargs.length) || 17)
      : r.hair === "styleMessy" ? ((r.hairargs && r.hairargs.length) || 9) : 0;
    for (var y = 11; y <= len; y++) { setpx(buf, HX0 - 1, y, base); setpx(buf, HX0, y, base); setpx(buf, HX1, y, base); setpx(buf, HX1 + 1, y, base); }
    for (var y2 = 4; y2 <= 12; y2++) { setpx(buf, 4, y2, sh); setpx(buf, 13, y2, sh); }
    [[5, 3], [12, 3], [5, 13], [12, 13], [6, 14], [11, 14]].forEach(function (p) { setpx(buf, p[0], p[1], sh); });
    [[7, 2], [8, 2], [9, 2], [10, 2], [7, 3], [8, 3], [9, 3]].forEach(function (p) { setpx(buf, p[0], p[1], hi); });
    for (var y3 = 4; y3 <= 11; y3++) setpx(buf, 9, y3, hi);
    for (var y4 = 4; y4 <= 12; y4++) setpx(buf, 8, y4, sh);
    rect(buf, 7, 14, 10, 14, sh);
    rect(buf, 7, 15, 10, 17, s.sh);
    rect(buf, 7, 15, 9, 15, s.base);
  }

  function drawHeadGroup(buf, r) {
    var skinBase = SKIN[r.skin].base;
    drawHead(buf, r.skin);
    drawFace(buf, r.skin, r.brow || "flat", r.mouth || "neutral", r.blush || false, r.lashes || false);
    HAIR[r.hair](buf, r.hairc, skinBase, r.hairargs || {});
  }

  function outlinePass(buf) {
    var pts = [];
    for (var y = 0; y < SCENE_H; y++) {
      for (var x = 0; x < SCENE_W; x++) {
        if (alphaAt(buf, x, y) !== 0) continue;
        if (alphaAt(buf, x + 1, y) === 255 || alphaAt(buf, x - 1, y) === 255 ||
            alphaAt(buf, x, y + 1) === 255 || alphaAt(buf, x, y - 1) === 255) pts.push([x, y]);
      }
    }
    pts.forEach(function (p) { setpx(buf, p[0], p[1], OUTLINE); });
  }

  function composeScene(r, phase, back) {
    var buf = new Uint8ClampedArray(SCENE_W * SCENE_H * 4);
    drawSceneTorso(buf, r, back);
    drawSceneLegs(buf, defaultPants(r), phase);
    if (back) drawHeadBack(buf, r); else drawHeadGroup(buf, r);
    outlinePass(buf);
    return buf;
  }

  // ─── recipes: Alex Chen (hedge fund manager), Marcus Webb (retail day
  // trader), Sara Kim (momentum trader) -- original designs, not a recolor of
  // any Office cast member. ────────────────────────────────────────────────
  var RECIPES = {
    Alex_Chen: {
      skin: "light", hairc: [40, 32, 26], hair: "styleShort", hairargs: { part: "L" },
      cloth: "suit", c1: [45, 52, 66], tie: [130, 45, 50], brow: "flat", mouth: "neutral"
    },
    Marcus_Webb: {
      skin: "tan", hairc: [70, 50, 30], hair: "styleMessy", hairargs: { length: 8 },
      cloth: "polo", c1: [90, 150, 170], c2: [70, 120, 140], brow: "raised", mouth: "smile"
    },
    Sara_Kim: {
      skin: "light", hairc: [30, 24, 22], hair: "styleFrame", hairargs: { length: 20, vol: 1 },
      cloth: "blouse", c1: [200, 90, 110], brow: "soft", mouth: "smile", lashes: true
    }
  };

  /** Blit an SCENE_W x SCENE_H RGBA buffer onto `ctx` at (dx, dy). */
  function blit(ctx, buf, dx, dy) {
    var img = ctx.createImageData(SCENE_W, SCENE_H);
    img.data.set(buf);
    var stage = document.createElement("canvas");
    stage.width = SCENE_W; stage.height = SCENE_H;
    stage.getContext("2d").putImageData(img, 0, 0);
    ctx.drawImage(stage, dx, dy);
  }

  /**
   * Build a Phaser texture for `underscoreKey` (e.g. "Alex_Chen") with the
   * SAME frame-name contract atlas.json used, so no other code in
   * main_script.html needs to change:
   *   idle frames:  "down" "up" "left" "right"
   *   walk frames:  "<dir>-walk.000".."003" for dir in down/up/left/right
   * `left` reuses the front (down) view, horizontally mirrored -- matching
   * Munder Difflin's own cast.ts, which does the same for its "right" row.
   * Returns true if built, false if no recipe exists for this key (caller
   * should fall back to the atlas-loaded sprite sheet in that case).
   */
  function build_cast_texture(scene, underscoreKey) {
    var r = RECIPES[underscoreKey];
    if (!r) return false;
    if (scene.textures.exists(underscoreKey)) return true; // already built this session

    var front = [composeScene(r, 0, false), composeScene(r, 1, false), composeScene(r, 2, false)];
    var back = [composeScene(r, 0, true), composeScene(r, 1, true), composeScene(r, 2, true)];

    // 6 columns (down x3, up x3) + 6 more for the mirrored left set = 12,
    // laid out in one row; "right" reuses the "down" columns directly.
    var cols = 12;
    var canvas = document.createElement("canvas");
    canvas.width = cols * SCENE_W; canvas.height = SCENE_H;
    var ctx = canvas.getContext("2d");
    ctx.imageSmoothingEnabled = false;

    for (var i = 0; i < 3; i++) blit(ctx, front[i], i * SCENE_W, 0);
    for (var j = 0; j < 3; j++) blit(ctx, back[j], (3 + j) * SCENE_W, 0);
    // Mirrored copies for "left" (columns 6-8).
    ctx.save();
    for (var k = 0; k < 3; k++) {
      ctx.save();
      ctx.translate((6 + k) * SCENE_W + SCENE_W, 0);
      ctx.scale(-1, 1);
      blit(ctx, front[k], 0, 0);
      ctx.restore();
    }
    ctx.restore();

    var texture = scene.textures.addCanvas(underscoreKey, canvas);
    var frameAt = function (name, col) { texture.add(name, 0, col * SCENE_W, 0, SCENE_W, SCENE_H); };
    // idle
    frameAt("down", 0); frameAt("up", 3); frameAt("right", 0); frameAt("left", 6);
    // walk cycles: stepL, stand, stepR, stand (4 frames from 3 poses)
    var walk = function (prefix, standCol, stepLCol, stepRCol) {
      frameAt(prefix + "-walk.000", stepLCol);
      frameAt(prefix + "-walk.001", standCol);
      frameAt(prefix + "-walk.002", stepRCol);
      frameAt(prefix + "-walk.003", standCol);
    };
    walk("down", 0, 1, 2);
    walk("up", 3, 4, 5);
    walk("right", 0, 1, 2);
    walk("left", 6, 7, 8);
    return true;
  }

  global.ProceduralCast = {
    RECIPE_NAMES: Object.keys(RECIPES),
    build_cast_texture: build_cast_texture
  };
})(window);
