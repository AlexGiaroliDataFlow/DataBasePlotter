(() => {
  "use strict";

  const canvas = document.getElementById("game");
  const ctx = canvas.getContext("2d");

  const ui = {
    overlay: document.getElementById("overlay"),
    overlayText: document.getElementById("overlayText"),
    startButton: document.getElementById("startButton"),
    phaseLabel: document.getElementById("phaseLabel"),
    bossName: document.getElementById("bossName"),
    score: document.getElementById("score"),
    integrityMeter: document.getElementById("integrityMeter"),
    bossMeter: document.getElementById("bossMeter"),
    flowMeter: document.getElementById("flowMeter"),
    abilityButtons: [...document.querySelectorAll("[data-ability]")]
  };

  const TAU = Math.PI * 2;
  const ARENA = { width: 1280, height: 720 };
  const input = {
    keys: new Set(),
    pointer: { x: ARENA.width * 0.42, y: ARENA.height * 0.5, down: false, active: false }
  };

  const bossPortrait = new Image();
  bossPortrait.src = "assets/boss.jpeg";

  const bosses = [
    {
      key: "photo",
      name: "Data-Flow Boss",
      maxHp: 430,
      hp: 430,
      color: "#34d6df",
      accent: "#73dd7d",
      desc: "A video-call boss that bends clean routes into loops.",
      pattern: 0
    }
  ];

  const state = {
    mode: "intro",
    time: 0,
    lastTime: 0,
    bossIndex: 0,
    boss: cloneBoss(0),
    player: {
      x: 170,
      y: ARENA.height / 2,
      radius: 18,
      hp: 100,
      invuln: 0,
      fireCooldown: 0,
      shield: 0,
      heat: 0
    },
    flow: 18,
    score: 0,
    combo: 1,
    shake: 0,
    spawnTimer: 0,
    bossTimer: 0,
    explosionTimer: 0,
    projectiles: [],
    enemies: [],
    particles: [],
    lanes: [],
    abilities: {
      burst: { cooldown: 0, max: 7 },
      shield: { cooldown: 0, max: 10 },
      reroute: { cooldown: 0, max: 8 }
    },
    message: "Pipeline armed"
  };

  function cloneBoss(index) {
    return { ...bosses[index], x: 1040, y: ARENA.height / 2, t: 0, shield: 0 };
  }

  function resetGame() {
    state.mode = "play";
    state.time = 0;
    state.lastTime = performance.now();
    state.bossIndex = 0;
    state.boss = cloneBoss(0);
    state.player.x = 170;
    state.player.y = ARENA.height / 2;
    state.player.hp = 100;
    state.player.invuln = 0;
    state.player.shield = 0;
    state.player.heat = 0;
    state.flow = 18;
    state.score = 0;
    state.combo = 1;
    state.shake = 0;
    state.spawnTimer = 0;
    state.bossTimer = 0;
    state.explosionTimer = 0;
    state.projectiles.length = 0;
    state.enemies.length = 0;
    state.particles.length = 0;
    state.abilities.burst.cooldown = 0;
    state.abilities.shield.cooldown = 0;
    state.abilities.reroute.cooldown = 0;
    state.message = bosses[0].desc;
    ui.overlay.classList.add("hidden");
    updateHud();
  }

  function resize() {
    const rect = canvas.getBoundingClientRect();
    const scale = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.max(1, Math.floor(rect.width * scale));
    canvas.height = Math.max(1, Math.floor(rect.height * scale));
    ctx.setTransform(canvas.width / ARENA.width, 0, 0, canvas.height / ARENA.height, 0, 0);
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function dist(a, b) {
    const dx = a.x - b.x;
    const dy = a.y - b.y;
    return Math.hypot(dx, dy);
  }

  function aimVector() {
    const p = state.player;
    const target = input.pointer.active ? input.pointer : { x: ARENA.width, y: p.y };
    const dx = target.x - p.x;
    const dy = target.y - p.y;
    const len = Math.hypot(dx, dy) || 1;
    return { x: dx / len, y: dy / len };
  }

  function spawnParticle(x, y, color, count = 6, speed = 110) {
    for (let i = 0; i < count; i += 1) {
      const a = Math.random() * TAU;
      const s = speed * (0.35 + Math.random());
      state.particles.push({
        x,
        y,
        vx: Math.cos(a) * s,
        vy: Math.sin(a) * s,
        life: 0.35 + Math.random() * 0.45,
        maxLife: 0.8,
        color,
        size: 2 + Math.random() * 4
      });
    }
  }

  function spawnProjectile(x, y, vx, vy, owner, kind, damage, radius, color) {
    state.projectiles.push({ x, y, vx, vy, owner, kind, damage, radius, color, life: 4 });
  }

  function spawnEnemy(kind, x, y) {
    const isMine = kind === "mine";
    state.enemies.push({
      kind,
      x,
      y,
      radius: isMine ? 22 : 16,
      hp: isMine ? 44 : 28,
      maxHp: isMine ? 44 : 28,
      vx: isMine ? -42 : -76,
      vy: isMine ? Math.sin(state.time + y) * 30 : 0,
      t: 0,
      color: isMine ? "#ffcd67" : "#ff6b73"
    });
  }

  function firePrimary() {
    const p = state.player;
    if (p.fireCooldown > 0 || p.heat > 96) return;
    const aim = aimVector();
    spawnProjectile(p.x + aim.x * 22, p.y + aim.y * 22, aim.x * 640, aim.y * 640, "player", "clean", 17, 6, "#7fffd4");
    p.fireCooldown = 0.13;
    p.heat = clamp(p.heat + 4.5, 0, 100);
  }

  function useAbility(name) {
    const ability = state.abilities[name];
    if (!ability || ability.cooldown > 0 || state.mode !== "play") return;

    if (name === "burst") {
      for (let i = 0; i < 14; i += 1) {
        const spread = -0.55 + i * (1.1 / 13);
        spawnProjectile(state.player.x + 24, state.player.y, 650, spread * 360, "player", "burst", 21, 7, "#34d6df");
      }
      spawnParticle(state.player.x, state.player.y, "#34d6df", 24, 230);
      state.message = "Compressed packets released";
    }

    if (name === "shield") {
      state.player.shield = 3.6;
      spawnParticle(state.player.x, state.player.y, "#73dd7d", 20, 120);
      state.message = "Firewall envelope online";
    }

    if (name === "reroute") {
      state.projectiles.push({
        x: state.player.x,
        y: state.player.y,
        vx: 880,
        vy: 0,
        owner: "player",
        kind: "reroute",
        damage: 54,
        radius: 18,
        color: "#ffcd67",
        life: 0.9
      });
      state.shake = 0.12;
      state.message = "Reroute pulse crossing lanes";
    }

    ability.cooldown = ability.max;
  }

  function update(dt) {
    state.time += dt;

    if (state.mode === "exploding") {
      state.explosionTimer -= dt;
      state.shake = Math.max(0, state.explosionTimer * 0.34);
      updateParticles(dt);
      updateHud();
      if (state.explosionTimer <= 0) endGame(true);
      return;
    }

    if (state.mode !== "play") return;

    state.bossTimer += dt;
    state.spawnTimer -= dt;
    state.shake = Math.max(0, state.shake - dt);
    state.player.fireCooldown = Math.max(0, state.player.fireCooldown - dt);
    state.player.invuln = Math.max(0, state.player.invuln - dt);
    state.player.shield = Math.max(0, state.player.shield - dt);
    state.player.heat = Math.max(0, state.player.heat - dt * 22);

    for (const ability of Object.values(state.abilities)) {
      ability.cooldown = Math.max(0, ability.cooldown - dt);
    }

    updatePlayer(dt);
    updateBoss(dt);
    updatePackets(dt);
    updateEnemies(dt);
    updateProjectiles(dt);
    updateParticles(dt);
    updateHud();

    if (state.player.hp <= 0 || state.flow <= 0) {
      endGame(false);
    }
  }

  function updatePlayer(dt) {
    const p = state.player;
    const speed = p.shield > 0 ? 225 : 285;
    let x = 0;
    let y = 0;

    if (input.keys.has("arrowleft") || input.keys.has("a")) x -= 1;
    if (input.keys.has("arrowright") || input.keys.has("d")) x += 1;
    if (input.keys.has("arrowup") || input.keys.has("w")) y -= 1;
    if (input.keys.has("arrowdown") || input.keys.has("s")) y += 1;

    if (input.pointer.down && input.pointer.active && input.pointer.x < ARENA.width * 0.62) {
      const dx = input.pointer.x - p.x;
      const dy = input.pointer.y - p.y;
      const len = Math.hypot(dx, dy);
      if (len > 12) {
        x += dx / len;
        y += dy / len;
      }
    }

    const len = Math.hypot(x, y) || 1;
    p.x = clamp(p.x + (x / len) * speed * dt, 42, ARENA.width * 0.58);
    p.y = clamp(p.y + (y / len) * speed * dt, 92, ARENA.height - 52);

    if (input.keys.has(" ") || input.keys.has("enter") || input.pointer.down) {
      firePrimary();
    }
  }

  function updateBoss(dt) {
    const b = state.boss;
    b.t += dt;
    b.y = ARENA.height / 2 + Math.sin(b.t * 0.9 + state.bossIndex) * 120;
    b.x = 1035 + Math.cos(b.t * 0.45) * 28;

    const cadence = [1.05, 0.95, 0.78][b.pattern];
    if (state.bossTimer >= cadence) {
      state.bossTimer = 0;
      if (b.pattern === 0) networkPattern();
      if (b.pattern === 1) modulePattern();
      if (b.pattern === 2) plushPattern();
    }

    if (b.pattern === 1) {
      b.shield = (Math.sin(b.t * 1.5) + 1) / 2;
    }
  }

  function networkPattern() {
    const b = state.boss;
    for (let i = -1; i <= 1; i += 1) {
      spawnProjectile(b.x - 72, b.y + i * 44, -285, i * 64, "boss", "latency", 11, 12, "#34d6df");
    }
    if (Math.random() < 0.65) spawnEnemy("glitch", ARENA.width - 80, 110 + Math.random() * 500);
  }

  function modulePattern() {
    const b = state.boss;
    const open = b.shield < 0.38;
    for (let i = 0; i < 6; i += 1) {
      const y = 120 + i * 86;
      const speed = open ? -330 : -230;
      spawnProjectile(b.x - 82, y, speed, Math.sin(i + b.t) * 36, "boss", "pin", 9, 8, open ? "#73dd7d" : "#ffcd67");
    }
    if (Math.random() < 0.55) spawnEnemy("mine", ARENA.width - 90, 130 + Math.random() * 470);
  }

  function plushPattern() {
    const b = state.boss;
    for (let i = 0; i < 5; i += 1) {
      const angle = Math.PI + (-0.55 + i * 0.275);
      spawnProjectile(b.x - 76, b.y, Math.cos(angle) * 270, Math.sin(angle) * 270, "boss", "thread", 10, 13, "#d4d9db");
    }
    const y = 98 + Math.floor(Math.random() * 6) * 92;
    spawnEnemy(Math.random() < 0.45 ? "mine" : "glitch", ARENA.width - 80, y);
  }

  function updatePackets(dt) {
    while (state.lanes.length < 18) {
      state.lanes.push({
        x: Math.random() * ARENA.width,
        y: 122 + Math.floor(Math.random() * 6) * 92,
        speed: 70 + Math.random() * 130,
        size: 4 + Math.random() * 4,
        clean: Math.random() > 0.18
      });
    }

    for (const packet of state.lanes) {
      packet.x += packet.speed * dt;
      if (packet.x > ARENA.width + 30) {
        packet.x = -30;
        packet.y = 122 + Math.floor(Math.random() * 6) * 92;
        packet.speed = 70 + Math.random() * 130;
        packet.clean = Math.random() > 0.2;
        if (packet.clean) {
          state.flow = clamp(state.flow + 0.8, 0, 100);
          state.score += 1;
        }
      }
    }

    state.flow = clamp(state.flow - dt * 1.3, 0, 100);
  }

  function updateEnemies(dt) {
    const p = state.player;
    for (let i = state.enemies.length - 1; i >= 0; i -= 1) {
      const enemy = state.enemies[i];
      enemy.t += dt;
      enemy.x += enemy.vx * dt;
      enemy.y += (enemy.vy + Math.sin(enemy.t * 4) * 18) * dt;

      if (enemy.kind === "mine") {
        enemy.vx -= dt * 9;
        enemy.radius = 20 + Math.sin(enemy.t * 5) * 2;
      }

      if (dist(enemy, p) < enemy.radius + p.radius) {
        damagePlayer(enemy.kind === "mine" ? 16 : 10);
        spawnParticle(enemy.x, enemy.y, enemy.color, 18, 190);
        state.enemies.splice(i, 1);
        continue;
      }

      if (enemy.x < -40) {
        state.flow = clamp(state.flow - 9, 0, 100);
        state.enemies.splice(i, 1);
      }
    }
  }

  function updateProjectiles(dt) {
    const p = state.player;
    const b = state.boss;

    for (let i = state.projectiles.length - 1; i >= 0; i -= 1) {
      const shot = state.projectiles[i];
      shot.x += shot.vx * dt;
      shot.y += shot.vy * dt;
      shot.life -= dt;

      if (shot.kind === "reroute") {
        shot.radius += dt * 44;
      }

      if (shot.owner === "player") {
        for (let e = state.enemies.length - 1; e >= 0; e -= 1) {
          const enemy = state.enemies[e];
          if (dist(shot, enemy) < shot.radius + enemy.radius) {
            enemy.hp -= shot.damage;
            spawnParticle(shot.x, shot.y, shot.color, 5, 120);
            if (shot.kind !== "reroute") state.projectiles.splice(i, 1);
            if (enemy.hp <= 0) {
              spawnParticle(enemy.x, enemy.y, enemy.color, 14, 180);
              state.score += 8;
              state.flow = clamp(state.flow + 2.5, 0, 100);
              state.enemies.splice(e, 1);
            }
            break;
          }
        }

        if (state.projectiles[i] !== shot) continue;

        const bossRadius = b.key === "photo" ? 104 : b.key === "plush" ? 88 : 72;
        if (dist(shot, b) < shot.radius + bossRadius) {
          const shielded = b.key === "module" && b.shield > 0.38;
          const damage = shielded ? shot.damage * 0.28 : shot.damage;
          b.hp = Math.max(0, b.hp - damage);
          state.score += Math.ceil(damage);
          state.flow = clamp(state.flow + damage * 0.025, 0, 100);
          state.shake = shielded ? 0.04 : 0.09;
          spawnParticle(shot.x, shot.y, shielded ? "#9aa6aa" : shot.color, shielded ? 5 : 11, shielded ? 90 : 170);
          if (shot.kind !== "reroute") state.projectiles.splice(i, 1);
          if (b.hp <= 0) {
            defeatBoss();
            return;
          }
        }
      } else if (dist(shot, p) < shot.radius + p.radius) {
        damagePlayer(shot.damage);
        spawnParticle(shot.x, shot.y, shot.color, 8, 130);
        state.projectiles.splice(i, 1);
        continue;
      }

      if (shot.life <= 0 || shot.x < -90 || shot.x > ARENA.width + 110 || shot.y < -100 || shot.y > ARENA.height + 100) {
        state.projectiles.splice(i, 1);
      }
    }
  }

  function updateParticles(dt) {
    for (let i = state.particles.length - 1; i >= 0; i -= 1) {
      const part = state.particles[i];
      part.life -= dt;
      part.x += part.vx * dt;
      part.y += part.vy * dt;
      part.vx *= 0.985;
      part.vy *= 0.985;
      if (part.life <= 0) state.particles.splice(i, 1);
    }
  }

  function damagePlayer(amount) {
    const p = state.player;
    if (p.invuln > 0) return;
    const blocked = p.shield > 0;
    const finalDamage = blocked ? amount * 0.28 : amount;
    p.hp = clamp(p.hp - finalDamage, 0, 100);
    p.invuln = blocked ? 0.1 : 0.36;
    state.shake = blocked ? 0.05 : 0.16;
    state.message = blocked ? "Firewall absorbed corruption" : "Integrity hit";
  }

  function defeatBoss() {
    explodeBoss();
  }

  function explodeBoss() {
    const b = state.boss;
    state.mode = "exploding";
    b.hp = 0;
    state.explosionTimer = 1.85;
    state.projectiles.length = 0;
    state.enemies.length = 0;
    state.message = "Boss overload: explosion";
    state.score += 240;
    state.flow = clamp(state.flow + 36, 0, 100);
    state.shake = 0.85;

    for (let ring = 0; ring < 6; ring += 1) {
      spawnParticle(b.x, b.y, ring % 2 ? "#ffcd67" : "#34d6df", 42, 260 + ring * 95);
      spawnParticle(b.x + (Math.random() - 0.5) * 135, b.y + (Math.random() - 0.5) * 120, "#ff6b73", 24, 230 + ring * 70);
    }
  }

  function endGame(won) {
    state.mode = won ? "won" : "lost";
    ui.overlay.classList.remove("hidden");
    ui.startButton.textContent = won ? "Run Again" : "Retry";
    ui.overlayText.textContent = won
      ? `Pipeline stabilized with ${Math.floor(state.score)} clean packets routed.`
      : "The stream collapsed. Reboot the pipeline and keep corruption away from the source.";
    updateHud();
  }

  function render() {
    ctx.save();
    ctx.clearRect(0, 0, ARENA.width, ARENA.height);

    if (state.shake > 0) {
      const s = state.shake * 18;
      ctx.translate((Math.random() - 0.5) * s, (Math.random() - 0.5) * s);
    }

    drawBackground();
    drawPackets();
    drawBoss();
    drawEnemies();
    drawProjectiles();
    drawPlayer();
    drawParticles();
    drawForeground();
    ctx.restore();
  }

  function drawBackground() {
    const grad = ctx.createLinearGradient(0, 0, ARENA.width, ARENA.height);
    grad.addColorStop(0, "#07161e");
    grad.addColorStop(0.55, "#0c2530");
    grad.addColorStop(1, "#10191e");
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, ARENA.width, ARENA.height);

    ctx.strokeStyle = "rgba(97, 204, 203, 0.12)";
    ctx.lineWidth = 1;
    for (let x = 0; x < ARENA.width; x += 64) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x + Math.sin(state.time + x) * 12, ARENA.height);
      ctx.stroke();
    }

    for (let i = 0; i < 6; i += 1) {
      const y = 122 + i * 92;
      ctx.strokeStyle = "rgba(115, 221, 125, 0.18)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(0, y);
      for (let x = 0; x <= ARENA.width; x += 80) {
        ctx.lineTo(x, y + Math.sin(state.time * 1.6 + x * 0.01 + i) * 8);
      }
      ctx.stroke();
    }

    ctx.fillStyle = "rgba(52, 214, 223, 0.09)";
    ctx.fillRect(64, 86, 82, ARENA.height - 128);
    ctx.fillStyle = "rgba(255, 205, 103, 0.1)";
    ctx.fillRect(1130, 86, 72, ARENA.height - 128);
  }

  function drawPackets() {
    for (const packet of state.lanes) {
      ctx.fillStyle = packet.clean ? "#73dd7d" : "#ff6b73";
      ctx.globalAlpha = packet.clean ? 0.72 : 0.92;
      roundedRect(packet.x, packet.y - packet.size / 2, packet.size * 4, packet.size, 3);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  }

  function drawPlayer() {
    const p = state.player;
    ctx.save();
    ctx.translate(p.x, p.y);

    if (p.shield > 0) {
      ctx.strokeStyle = "rgba(115, 221, 125, 0.74)";
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.arc(0, 0, 31 + Math.sin(state.time * 14) * 3, 0, TAU);
      ctx.stroke();
    }

    const aim = aimVector();
    ctx.rotate(Math.atan2(aim.y, aim.x));
    ctx.fillStyle = p.invuln > 0 ? "#ffffff" : "#e9fbff";
    ctx.strokeStyle = "#34d6df";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(27, 0);
    ctx.lineTo(-15, -18);
    ctx.lineTo(-8, 0);
    ctx.lineTo(-15, 18);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = "#0b242c";
    roundedRect(-12, -8, 22, 16, 4);
    ctx.fill();
    ctx.restore();
  }

  function drawBoss() {
    const b = state.boss;
    ctx.save();
    ctx.translate(b.x, b.y);

    if (b.key === "photo") drawPhotoBoss(b);
    if (b.key === "network") drawNetworkBoss(b);
    if (b.key === "module") drawModuleBoss(b);
    if (b.key === "plush") drawPlushBoss(b);

    ctx.restore();
  }

  function drawPhotoBoss(b) {
    const exploding = state.mode === "exploding";
    const pulse = Math.sin(b.t * 2.2) * 5;
    const width = 210 + pulse;
    const height = 142 + pulse;

    if (exploding) {
      const flash = clamp(state.explosionTimer / 1.85, 0, 1);
      ctx.fillStyle = `rgba(255, 205, 103, ${0.22 + flash * 0.28})`;
      ctx.beginPath();
      ctx.arc(0, 0, 158 + (1 - flash) * 140, 0, TAU);
      ctx.fill();
    }

    ctx.fillStyle = "rgba(52, 214, 223, 0.16)";
    ctx.beginPath();
    ctx.arc(0, 0, 132 + Math.sin(state.time * 4) * 8, 0, TAU);
    ctx.fill();

    ctx.save();
    roundedRect(-width / 2, -height / 2, width, height, 18);
    ctx.clip();

    if (bossPortrait.complete && bossPortrait.naturalWidth > 0) {
      drawImageCover(bossPortrait, -width / 2, -height / 2, width, height);
    } else {
      const fallback = ctx.createLinearGradient(-width / 2, -height / 2, width / 2, height / 2);
      fallback.addColorStop(0, "#b7dde7");
      fallback.addColorStop(1, "#0d4252");
      ctx.fillStyle = fallback;
      ctx.fillRect(-width / 2, -height / 2, width, height);
    }

    if (exploding) {
      ctx.globalCompositeOperation = "screen";
      ctx.fillStyle = `rgba(255, 255, 255, ${0.18 + Math.random() * 0.24})`;
      ctx.fillRect(-width / 2, -height / 2, width, height);
    }
    ctx.restore();

    ctx.strokeStyle = exploding ? "#ffcd67" : "#34d6df";
    ctx.lineWidth = exploding ? 8 : 5;
    roundedRect(-width / 2, -height / 2, width, height, 18);
    ctx.stroke();

    ctx.strokeStyle = "rgba(115, 221, 125, 0.58)";
    ctx.lineWidth = 3;
    for (let i = 0; i < 8; i += 1) {
      const a = i * TAU / 8 + state.time * 0.7;
      const x = Math.cos(a) * 132;
      const y = Math.sin(a) * 132;
      ctx.beginPath();
      ctx.moveTo(Math.cos(a) * 92, Math.sin(a) * 76);
      ctx.lineTo(x, y);
      ctx.stroke();
      ctx.fillStyle = i % 2 ? "#73dd7d" : "#34d6df";
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, TAU);
      ctx.fill();
    }
  }

  function drawImageCover(image, x, y, width, height) {
    const scale = Math.max(width / image.naturalWidth, height / image.naturalHeight);
    const sw = width / scale;
    const sh = height / scale;
    const sx = (image.naturalWidth - sw) / 2;
    const sy = (image.naturalHeight - sh) / 2;
    ctx.drawImage(image, sx, sy, sw, sh, x, y, width, height);
  }

  function drawNetworkBoss(b) {
    ctx.fillStyle = "rgba(52, 214, 223, 0.12)";
    ctx.beginPath();
    ctx.arc(0, 0, 108 + Math.sin(b.t * 2) * 8, 0, TAU);
    ctx.fill();

    ctx.strokeStyle = b.color;
    ctx.lineWidth = 5;
    ctx.beginPath();
    ctx.arc(0, 0, 72, -0.9, TAU - 0.9);
    ctx.stroke();

    ctx.fillStyle = "#071218";
    ctx.beginPath();
    ctx.arc(-15, -10, 7, 0, TAU);
    ctx.arc(15, -10, 7, 0, TAU);
    ctx.fill();

    ctx.strokeStyle = "#73dd7d";
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.arc(0, 3, 30, 0.18, Math.PI - 0.18);
    ctx.stroke();

    for (let i = 0; i < 9; i += 1) {
      const a = i * TAU / 9 + b.t * 0.8;
      const x = Math.cos(a) * 100;
      const y = Math.sin(a) * 100;
      ctx.fillStyle = i % 2 ? "#73dd7d" : "#34d6df";
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, TAU);
      ctx.fill();
      ctx.strokeStyle = "rgba(157, 221, 220, 0.35)";
      ctx.beginPath();
      ctx.moveTo(x * 0.52, y * 0.52);
      ctx.lineTo(x, y);
      ctx.stroke();
    }
  }

  function drawModuleBoss(b) {
    ctx.fillStyle = "#1a2529";
    roundedRect(-88, -70, 176, 140, 14);
    ctx.fill();
    ctx.strokeStyle = "#9aa6aa";
    ctx.lineWidth = 8;
    roundedRect(-88, -70, 176, 140, 14);
    ctx.stroke();

    ctx.fillStyle = "#164b3b";
    roundedRect(-64, -46, 128, 92, 6);
    ctx.fill();

    ctx.fillStyle = "#0b171b";
    roundedRect(-20, -18, 42, 38, 4);
    ctx.fill();

    ctx.fillStyle = "#b8c2c4";
    for (let i = 0; i < 8; i += 1) {
      roundedRect(-70 + i * 20, -76, 10, 16, 3);
      ctx.fill();
      roundedRect(-70 + i * 20, 60, 10, 16, 3);
      ctx.fill();
    }

    const shieldAlpha = 0.18 + b.shield * 0.52;
    ctx.strokeStyle = `rgba(255, 205, 103, ${shieldAlpha})`;
    ctx.lineWidth = 6;
    ctx.beginPath();
    ctx.arc(0, 0, 112, 0, TAU);
    ctx.stroke();
  }

  function drawPlushBoss(b) {
    ctx.fillStyle = "#42c46d";
    roundedRect(-112, -78, 224, 156, 24);
    ctx.fill();
    ctx.strokeStyle = "#93efa1";
    ctx.lineWidth = 5;
    roundedRect(-112, -78, 224, 156, 24);
    ctx.stroke();

    ctx.strokeStyle = "rgba(217, 255, 197, 0.58)";
    ctx.lineWidth = 4;
    for (let i = -2; i <= 2; i += 1) {
      ctx.beginPath();
      ctx.moveTo(-88, i * 25);
      ctx.lineTo(-15, i * 10);
      ctx.lineTo(76, i * 22);
      ctx.stroke();
    }

    ctx.fillStyle = "#151d20";
    roundedRect(15, -22, 62, 48, 5);
    ctx.fill();

    ctx.fillStyle = "#d4d9db";
    for (let i = 0; i < 9; i += 1) {
      roundedRect(-102 + i * 23, 52, 16, 34, 5);
      ctx.fill();
    }

    ctx.fillStyle = "rgba(255,255,255,0.24)";
    ctx.beginPath();
    ctx.ellipse(-46, -42, 22, 10, -0.24, 0, TAU);
    ctx.fill();
  }

  function drawEnemies() {
    for (const enemy of state.enemies) {
      ctx.save();
      ctx.translate(enemy.x, enemy.y);
      ctx.rotate(enemy.t * 1.8);
      ctx.fillStyle = enemy.color;
      if (enemy.kind === "mine") {
        for (let i = 0; i < 8; i += 1) {
          ctx.rotate(TAU / 8);
          roundedRect(8, -4, 18, 8, 2);
          ctx.fill();
        }
        ctx.beginPath();
        ctx.arc(0, 0, 18, 0, TAU);
        ctx.fill();
      } else {
        roundedRect(-14, -14, 28, 28, 4);
        ctx.fill();
        ctx.fillStyle = "#071218";
        roundedRect(-5, -11, 10, 22, 2);
        ctx.fill();
      }
      ctx.restore();
    }
  }

  function drawProjectiles() {
    for (const shot of state.projectiles) {
      ctx.save();
      ctx.translate(shot.x, shot.y);
      ctx.fillStyle = shot.color;
      ctx.strokeStyle = shot.color;
      ctx.lineWidth = 3;

      if (shot.kind === "reroute") {
        ctx.globalAlpha = clamp(shot.life, 0, 0.9);
        ctx.beginPath();
        ctx.moveTo(-8, -shot.radius);
        ctx.lineTo(72, -shot.radius * 0.45);
        ctx.lineTo(120, 0);
        ctx.lineTo(72, shot.radius * 0.45);
        ctx.lineTo(-8, shot.radius);
        ctx.stroke();
      } else {
        ctx.beginPath();
        ctx.arc(0, 0, shot.radius, 0, TAU);
        ctx.fill();
        ctx.globalAlpha = 0.32;
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.lineTo(-shot.vx * 0.045, -shot.vy * 0.045);
        ctx.stroke();
      }
      ctx.restore();
    }
    ctx.globalAlpha = 1;
  }

  function drawParticles() {
    for (const part of state.particles) {
      ctx.globalAlpha = clamp(part.life / part.maxLife, 0, 1);
      ctx.fillStyle = part.color;
      ctx.beginPath();
      ctx.arc(part.x, part.y, part.size, 0, TAU);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  }

  function drawForeground() {
    ctx.strokeStyle = "rgba(233, 251, 255, 0.16)";
    ctx.lineWidth = 2;
    roundedRect(64, 86, ARENA.width - 128, ARENA.height - 128, 12);
    ctx.stroke();
  }

  function roundedRect(x, y, width, height, radius) {
    const r = Math.min(radius, width / 2, height / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + width, y, x + width, y + height, r);
    ctx.arcTo(x + width, y + height, x, y + height, r);
    ctx.arcTo(x, y + height, x, y, r);
    ctx.arcTo(x, y, x + width, y, r);
    ctx.closePath();
  }

  function updateHud() {
    const b = state.boss;
    ui.integrityMeter.value = Math.round(state.player.hp);
    ui.bossMeter.value = b ? Math.round((b.hp / b.maxHp) * 100) : 0;
    ui.flowMeter.value = Math.round(state.flow);
    ui.phaseLabel.textContent = state.message;
    ui.bossName.textContent = b ? b.name : "Pipeline clear";
    ui.score.textContent = `${Math.floor(state.score)} clean packets`;

    for (const button of ui.abilityButtons) {
      const ability = state.abilities[button.dataset.ability];
      button.disabled = ability.cooldown > 0 || state.mode !== "play";
      const label = button.querySelector("strong");
      if (ability.cooldown > 0) label.textContent = `${Math.ceil(ability.cooldown)}s`;
      else if (button.dataset.ability === "burst") label.textContent = "Burst";
      else if (button.dataset.ability === "shield") label.textContent = "Shield";
      else label.textContent = "Reroute";
    }
  }

  function loop(now) {
    const dt = Math.min((now - state.lastTime) / 1000 || 0, 0.033);
    state.lastTime = now;
    update(dt);
    render();
    requestAnimationFrame(loop);
  }

  function pointerToArena(event) {
    const rect = canvas.getBoundingClientRect();
    input.pointer.x = ((event.clientX - rect.left) / rect.width) * ARENA.width;
    input.pointer.y = ((event.clientY - rect.top) / rect.height) * ARENA.height;
    input.pointer.active = true;
  }

  window.addEventListener("resize", resize);
  window.addEventListener("keydown", (event) => {
    const key = event.key.toLowerCase();
    input.keys.add(key);
    if ([" ", "arrowup", "arrowdown", "arrowleft", "arrowright"].includes(key)) event.preventDefault();
    if (key === "1") useAbility("burst");
    if (key === "2") useAbility("shield");
    if (key === "3") useAbility("reroute");
    if (key === "p") {
      state.mode = state.mode === "play" ? "pause" : "play";
      state.message = state.mode === "pause" ? "Paused" : "Pipeline running";
      updateHud();
    }
  });
  window.addEventListener("keyup", (event) => input.keys.delete(event.key.toLowerCase()));

  canvas.addEventListener("pointerdown", (event) => {
    pointerToArena(event);
    input.pointer.down = true;
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", pointerToArena);
  canvas.addEventListener("pointerup", () => {
    input.pointer.down = false;
  });
  canvas.addEventListener("pointercancel", () => {
    input.pointer.down = false;
  });

  ui.startButton.addEventListener("click", resetGame);
  for (const button of ui.abilityButtons) {
    button.addEventListener("click", () => useAbility(button.dataset.ability));
  }

  resize();
  updateHud();
  requestAnimationFrame((now) => {
    state.lastTime = now;
    requestAnimationFrame(loop);
  });
})();
