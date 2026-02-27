---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
title: ""
permalink: /
---

<canvas id="bg-canvas"></canvas>

<div style="padding: 10px 0; max-width: 800px;">
  <div style="display: flex; align-items: center; gap: 20px; margin-bottom: 20px;">
    <div style="flex: 1; text-align: center;">
      <img src="/assets/images/me-welcome.jpg" alt="Abhishek Sharma" style="max-width: 140px; border-radius: 50%; box-shadow: 0 10px 30px rgba(0,0,0,0.1);">
    </div>
    <div style="flex: 2;">
      <h1 style="font-size: 2.2rem; margin-bottom: 5px;">Abhishek Sharma</h1>
      <p style="font-size: 1.1rem; color: #666; font-weight: 300; margin-bottom: 15px;">
        Architecting Backend Systems · Driving AI Integration
      </p>
      <div class="social-links">
        <a href="https://github.com/avisekksarma" target="_blank" title="GitHub"><i class="fa-brands fa-github"></i></a>
        <a href="https://linkedin.com/in/aviseksarma" target="_blank" title="LinkedIn"><i class="fa-brands fa-linkedin"></i></a>
      </div>
    </div>
  </div>

  <section style="margin-bottom: 25px; line-height: 1.6;">
    <p style="font-size: 1.05rem;">
      I work on architecting robust backend infrastructures and leveraging AI-driven automation in production. I am particularly interested in how scalable system design can be used to deploy and maintain intelligent services at scale, along with integrating AI to solve real-world problems.
    </p>
  </section>

  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
    <div class="card">
      <h3 style="margin-bottom: 10px;">🚀 Current Focus</h3>
      <p>Building an AI-powered compliance automation platform at <strong>ZeroTB</strong>. Architecting real-time audit engines and scalable backend services.</p>
    </div>
    <div class="card">
      <h3 style="margin-bottom: 10px;">📄 Research</h3>
      <p>Passionate about NLP for low-resource languages. Published research on <strong>Nepali Topic Modeling</strong> in IEEE (ICICT 2025).</p>
    </div>
  </div>

  <div style="margin-top: 30px; border-top: 1px solid #eee; padding-top: 20px;">
    <p style="text-align: center; color: #888; font-style: italic; font-size: 0.95rem;">
      Explore my <a href="/experience/">Professional Path</a>, view my <a href="/projects/">Impactful Projects</a>, or <a href="/about/">Get to know me</a>.
    </p>
  </div>
</div>

<script>
  (function() {
    const canvas = document.getElementById('bg-canvas');
    const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.z = 250;

    const particlesCount = 100;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(particlesCount * 3);
    const velocities = new Float32Array(particlesCount * 3);

    for (let i = 0; i < particlesCount * 3; i++) {
      positions[i] = (Math.random() - 0.5) * 600;
      velocities[i] = (Math.random() - 0.5) * 0.2;
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const material = new THREE.PointsMaterial({ size: 2, color: 0x00FF00, transparent: true, opacity: 0.8 });
    const points = new THREE.Points(geometry, material);
    scene.add(points);

    // Lines
    const lineMaterial = new THREE.LineBasicMaterial({ color: 0x00FF00, transparent: true, opacity: 0.35 });
    let linesMesh;

    function createLines() {
      if (linesMesh) scene.remove(linesMesh);
      const linePositions = [];
      const currentPos = geometry.attributes.position.array;

      for (let i = 0; i < particlesCount; i++) {
        for (let j = i + 1; j < particlesCount; j++) {
          const dx = currentPos[i * 3] - currentPos[j * 3];
          const dy = currentPos[i * 3 + 1] - currentPos[j * 3 + 1];
          const dz = currentPos[i * 3 + 2] - currentPos[j * 3 + 2];
          const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

          if (dist < 80) {
            linePositions.push(currentPos[i * 3], currentPos[i * 3 + 1], currentPos[i * 3 + 2]);
            linePositions.push(currentPos[j * 3], currentPos[j * 3 + 1], currentPos[j * 3 + 2]);
          }
        }
      }

      const lineGeometry = new THREE.BufferGeometry();
      lineGeometry.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3));
      linesMesh = new THREE.LineSegments(lineGeometry, lineMaterial);
      scene.add(linesMesh);
    }

    let mouseX = 0, mouseY = 0;
    document.addEventListener('mousemove', (e) => {
      mouseX = (e.clientX - window.innerWidth / 2) * 0.05;
      mouseY = (e.clientY - window.innerHeight / 2) * 0.05;
    });

    function animate() {
      requestAnimationFrame(animate);
      
      const pos = geometry.attributes.position.array;
      for (let i = 0; i < particlesCount * 3; i++) {
        pos[i] += velocities[i];
        if (pos[i] > 300 || pos[i] < -300) velocities[i] *= -1;
      }
      geometry.attributes.position.needsUpdate = true;
      
      createLines();

      camera.position.x += (mouseX - camera.position.x) * 0.05;
      camera.position.y += (-mouseY - camera.position.y) * 0.05;
      camera.lookAt(scene.position);

      renderer.render(scene, camera);
    }

    animate();

    window.addEventListener('resize', () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    });
  })();
</script>
