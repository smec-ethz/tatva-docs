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
</style>

# 

<div align="center">

<img src="assets/logo-small.png" alt="drawing" width="300" height="60"/>

<h3 align="center">tatva: Lego-like building blocks for differentiable FEM</h3>

</div>

`tatva` (तत्त्व) is a Sanskrit word meaning _principle_ or _elements of reality_.
True to its name, `tatva` provides fundamental Lego-like building blocks
(elements) which can be used to construct complex finite element method (FEM)
simulations as energy functionals. `tatva` is a pure Python library for FEM simulations and is
built on top of JAX ecosystem, making it easy to use FEM in a differentiable
way.

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
    <h3>Versitality</h3>
    <p> Operator abstractions that map, integrate, differentiate on arbitrary meshes. Capability to handle mixed-dimension coupling, multi-point constaints, and more. </p>
  </div>
  <div class="card">
    <h3>High Performance</h3>
    <p>Built-in sparse differentiation via coloring and matrix-free assembly tailored for mordern architecture such as GPUs. </p>
  </div>
</div>

<div style="width: 100%; margin: 20px 0;">
    <iframe src="https://www.linkedin.com/embed/feed/update/urn:li:ugcPost:7442949970275504128?collapsed=1" 
            height="600" 
            style="width: 100%; border: none;" 
            frameborder="0" 
            allowfullscreen="" 
            title="">
    </iframe>
</div>


<div style="width: 100%; margin: 20px 0;">
    <iframe src="https://www.linkedin.com/embed/feed/update/urn:li:ugcPost:7434187928412606464?collapsed=1"
            height="600" 
            style="width: 100%; border: none;" 
            frameborder="0" 
            allowfullscreen="" 
            title="">
    </iframe>
</div>


<div style="width: 100%; margin: 20px 0;">
    <iframe src="https://www.linkedin.com/embed/feed/update/urn:li:share:7429468306744586240?collapsed=1" 
            height="300" 
            style="width: 100%; border: none;" 
            frameborder="0" 
            allowfullscreen="" 
            title="">
    </iframe>
</div>
