---
hide:
  - navigation
  - toc
---

<style>
  /* Target the main content grid and ignore the header */
  main .md-grid {
    max-width: 60rem !important;
  }

  #__skip {
    display: none;
  }
</style>

<div class="hero">

<img src="assets/tatva.svg" class="logo" alt="drawing" />

<h3>Lego-like building blocks for differentiable FEM</h3>

<p>
<b>tatva (तत्त्व)</b> is a Sanskrit word meaning <i>principle</i> or <i>elements of reality</i>.
True to its name, <code>tatva</code> provides fundamental Lego-like building blocks
(elements) which can be used to construct complex finite element method (FEM)
simulations as energy functionals. <code>tatva</code> is a pure Python library for FEM simulations and is
built on top of JAX ecosystem, making it easy to use FEM in a differentiable
way.
</p>

<div class="hero-buttons">
<a href="getting_started" class="button">👉 Get Started</a>
<a href="examples/linear_elasticity" class="button">👷‍♀️ Examples</a>
</div>

</div>

<section id="home">

<div class="swiper">
<div class="swiper-wrapper">

<div class="swiper-slide">
  <img src="assets/images/contact_animation.gif" alt="Hertzian Contact">
  <div class="slide-caption">
    <h3>Hertzian Contact</h3>
  </div>
</div>

<div class="swiper-slide">
  <img src="assets/images/cohesive_fracture.gif" alt="Cohesive Fracture">
  <div class="slide-caption">
    <h3>Cohesive Fracture</h3>
  </div>
</div>

<div class="swiper-slide">
  <img src="assets/images/surface_advection_diffusion.gif" alt="Surface Advection Diffusion">
  <div class="slide-caption">
    <h3>Surface Advection Diffusion</h3>
  </div>
</div>

</div>

<div class="swiper-pagination"></div>

<div class="swiper-button-prev"></div>
<div class="swiper-button-next"></div>
</div>

<div class="grid-cards">
  <div class="card">
    <h3>Energy-Centric Solver</h3>
    <p>Energy-based formulation of FEM operators with automatic differentiation via JAX. Just write energy and differentiate it directly.</p>
  </div>
  <div class="card">
    <h3>Versatility</h3>
    <p> Operator abstractions that map, integrate, differentiate on arbitrary meshes. Capability to handle mixed-dimension coupling, multi-point constaints, and more. </p>
  </div>
  <div class="card">
    <h3>High Performance</h3>
    <p>Built-in sparse differentiation via coloring and matrix-free assembly tailored for mordern architecture such as GPUs. </p>
  </div>
</div>

</section>

<div class="social-embeds">

<h3>Find us on LinkedIn</h3>

<div class="social-container">

<iframe src="https://www.linkedin.com/embed/feed/update/urn:li:share:7469644702548426752?collapsed=1" height="470" width="504" frameborder="0" allowfullscreen=""></iframe>

<iframe src="https://www.linkedin.com/embed/feed/update/urn:li:ugcPost:7442949970275504128?collapsed=1" height="550" width="504" frameborder="0" allowfullscreen=""></iframe>

<iframe src="https://www.linkedin.com/embed/feed/update/urn:li:ugcPost:7434187928412606464?collapsed=1" height="550" width="504" frameborder="0" allowfullscreen=""></iframe>

<iframe src="https://www.linkedin.com/embed/feed/update/urn:li:share:7429468306744586240?collapsed=1" height="264" width="504" frameborder="0" allowfullscreen=""></iframe>

</div>

</div>
