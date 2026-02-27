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
    camera.position.set(0, 120, 180);
    camera.lookAt(0, 0, 0);

    // Flowing Topography Grid
    const size = 800;
    const divisions = 70;
    const geometry = new THREE.PlaneGeometry(size, size, divisions, divisions);
    const material = new THREE.PointsMaterial({ 
      color: 0x00FF00, 
      size: 1.0, 
      transparent: true, 
      opacity: 0.3 
    });
    
    const count = geometry.attributes.position.count;
    const grid = new THREE.Points(geometry, material);
    grid.rotation.x = -Math.PI / 2.2;
    scene.add(grid);

    // Interaction Tools
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2(-100, -100); // Start off-screen
    const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
    const pointOfIntersection = new THREE.Vector3();

    document.addEventListener('mousemove', (e) => {
      mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
      mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
    });

    let time = 0;
    function animate() {
      requestAnimationFrame(animate);
      time += 0.01;

      // Project mouse to 3D space
      raycaster.setFromCamera(mouse, camera);
      raycaster.ray.intersectPlane(plane, pointOfIntersection);
      
      // We need to account for grid rotation to find local intersection
      const localPoint = pointOfIntersection.clone().applyMatrix4(grid.matrixWorld.invert());

      const positions = geometry.attributes.position;
      for (let i = 0; i < count; i++) {
        const x = positions.getX(i);
        const y = positions.getY(i);
        
        // Base topography wave
        let z = Math.sin(x / 50 + time) * 12 + 
                Math.cos(y / 50 + time) * 12 + 
                Math.sin((x + y) / 80 + time) * 8;
        
        // Mouse displacement effect
        const dx = x - localPoint.x;
        const dy = y - localPoint.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 60) {
          z += (60 - dist) * 0.8; // "Push" effect
        }
        
        positions.setZ(i, z);
      }
      positions.needsUpdate = true;

      // Subtle interaction sway
      grid.rotation.z += (mouse.x * 0.05 - grid.rotation.z) * 0.05;
      grid.position.y += (-mouse.y * 10 - grid.position.y) * 0.05;

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
