

    document.addEventListener('DOMContentLoaded', async () => {
          console.log('it is woring....')

          const res = await fetch('https://api.ultimatejaipurians.in/upload/images')
                 
          const data = await res.json();

          // console.log(data);


          const track = document.getElementById('track-pro');
                track.innerHTML = '';
      let index = 0;
      data.images.forEach(img => {
      const card = document.createElement('div');
      card.className = 'production-carousel-card';

      card.dataset.image = img.imageUrl;

      card.innerHTML = `
        <div class="production-in-card">
          <span id=${img._id} class="cross-icon"> x </span>
          <div class="production-img">
            <img src="${img.imageUrl}" />
          </div>
        </div>
      `;

      track.appendChild(card);
    });




const images = track.querySelectorAll('img');

let loaded = 0;
images.forEach(img => {
  if (img.complete) {
    loaded++;
  } else {
    img.onload = () => {
      loaded++;
      if (loaded === images.length) {
        initProductionCarousel(); // 🔥 build AFTER images load
      }
    };
    img.onerror = () => {
      loaded++;
      if (loaded === images.length) {
        initProductionCarousel();
      }
    };
  }
});

// In case all images already cached
if (loaded === images.length) {
  initProductionCarousel();
}

          
});




  async function removePhoto(imgId, key){

    console.log('api called...')
   if(key === "Reunion"){
      
    const res = await fetch(`https://api.ultimatejaipurians.in/upload/delete?id=${imgId}`,
                              {
                                method: "DELETE",
                                headers: { 'Content-Type': 'application/json' },
                              })

    const data = await res.json();

    const {status} = data;

    if(!status){
      alert('Photo is not deleted. Try After Sometime.')
    } else {
      alert('Photo is deleted.')
      window.location.reload()
    }
    }
    else {
      alert('Your Pass Key is wrong')
    }
   }






  document.getElementById('track-pro').addEventListener('click', (e) => {

    // X button clicked logic

    if (e.target.classList.contains('cross-icon')) {

    const card = e.target.closest('.production-carousel-card');
    const imageId = e.target.id; 
    console.log('X clicked');
    console.log('Image ID:', imageId);


    const key = window.prompt('Enter your pass key.')


    removePhoto(imageId, key)

    return;
  
  }



    // Card clicked logic
  const card = e.target.closest('.production-carousel-card');
  if (!card) return;

  const imageUrl = card.dataset.image;
  if (!imageUrl) return;

  openPopPhoto(imageUrl);
});


function openPopPhoto(imageUrl) {
  const popup = document.getElementById('popup-photo');
  const img = document.getElementById('popup-photo-img');

  img.src = imageUrl;
  popup.style.display = 'flex';

  function closePopup() {
    document.getElementById("popup-photo").style.display = "none";
    }

  document.getElementById("popup-photo").addEventListener("click", (e) => {
   if (e.target.id === "popup-photo") closePopup();
  });

}




function initProductionCarousel() {

  const track = document.getElementById('track-pro');
  const stage = document.getElementById('stage-pro');
  const prevBtn = document.getElementById('prev-pro');
  const nextBtn = document.getElementById('next-pro');

  let originalHTML = [];
  let slideCount = 0;

  let visibleCount = getVisibleCount();
  let slides = [];
  let positionIndex = 0;
  const transitionMs = 450;
  let isTransitioning = false;
  let cardWidth = 0;
  let gap = 0;
  let autoSlide = null;

  function getVisibleCount() {
    if (window.matchMedia('(max-width:520px)').matches) return 1;
    if (window.matchMedia('(max-width:900px)').matches) return 2;
    return 3;
  }

  // 🔥 NEW: capture slides AFTER API loads
  function captureOriginalSlides() {
    originalHTML = Array.from(track.children).map(el => el.outerHTML);
    slideCount = originalHTML.length;
  }

  // 🔥 SAFE BUILD
  function build() {
    captureOriginalSlides();

    if (!slideCount) {
      console.warn('Carousel build skipped: no slides');
      return;
    }

    track.style.transition = 'none';
    track.innerHTML = '';

    visibleCount = getVisibleCount();

    const originals = originalHTML.map(html => {
      const tmp = document.createElement('div');
      tmp.innerHTML = html;
      return tmp.firstElementChild;
    });

    const frontClones = [];
    const endClones = [];

    for (let i = 0; i < visibleCount; i++) {
      const idxFront = (slideCount - visibleCount + i) % slideCount;

      frontClones.push(originals[idxFront].cloneNode(true));
      endClones.push(originals[i % slideCount].cloneNode(true));
    }

    frontClones.forEach(n => {
      n.classList.add('clone');
      track.appendChild(n);
    });

    originals.forEach(n => track.appendChild(n));

    endClones.forEach(n => {
      n.classList.add('clone');
      track.appendChild(n);
    });

    slides = Array.from(track.children);
    positionIndex = visibleCount;

    computeCardDimensions();
    updateTrack(false);

    requestAnimationFrame(() => {
      track.style.transition = `transform ${transitionMs}ms cubic-bezier(.25,.8,.25,1)`;
    });

    track.removeEventListener('transitionend', onTransitionEnd);
    track.addEventListener('transitionend', onTransitionEnd);
  }

  function computeCardDimensions() {
    const card = track.querySelector('.production-carousel-card');
    if (!card) return;

    cardWidth = card.getBoundingClientRect().width;
    gap = parseFloat(getComputedStyle(track).gap) || 30;
  }

  function updateTrack(animate = true) {
    track.style.transition = animate
      ? `transform ${transitionMs}ms cubic-bezier(.25,.8,.25,1)`
      : 'none';

    track.style.transform = `translateX(-${positionIndex * (cardWidth + gap)}px)`;
  }

  function onTransitionEnd() {
    isTransitioning = false;

    if (positionIndex >= visibleCount + slideCount) {
      positionIndex -= slideCount;
      updateTrack(false);
    }
    if (positionIndex < visibleCount) {
      positionIndex += slideCount;
      updateTrack(false);
    }
  }

  function next() {
    if (isTransitioning) return;
    isTransitioning = true;
    positionIndex++;
    updateTrack(true);
  }

  function prev() {
    if (isTransitioning) return;
    isTransitioning = true;
    positionIndex--;
    updateTrack(true);
  }

  prevBtn.addEventListener('click', prev);
  nextBtn.addEventListener('click', next);

  function startAuto() {
    stopAuto();
    autoSlide = setInterval(next, 2000);
  }

  function stopAuto() {
    if (autoSlide) clearInterval(autoSlide);
    autoSlide = null;
  }

  stage.addEventListener('mouseenter', stopAuto);
  stage.addEventListener('mouseleave', startAuto);

  window.addEventListener('resize', () => build());

  build();
  startAuto();


}
