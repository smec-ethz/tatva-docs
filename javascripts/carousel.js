// docs/javascripts/carousel.js
document.addEventListener("DOMContentLoaded", function () {
  const swiper = new Swiper(".swiper", {
    // Optional parameters
    direction: "horizontal",
    loop: true,
    speed: 600,
    autoplay: {
      delay: 5000, // 5 seconds per slide
      disableOnInteraction: false,
    },

    // If you want arrows
    navigation: {
      nextEl: ".swiper-button-next",
      prevEl: ".swiper-button-prev",
    },

    // If you want dots at the bottom
    pagination: {
      el: ".swiper-pagination",
      clickable: true,
    },
  });
});
