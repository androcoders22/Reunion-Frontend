

document.addEventListener('DOMContentLoaded', async () => {
  console.log('it is woring....');

  const track = document.getElementById('track-pro');
  track.innerHTML = '';

  // Keep local memories in strict order: image 1 -> image 19.
  const localImages = Array.from({ length: 19 }, (_, i) => ({
    imageUrl: `image 2/image ${i + 1}.jpeg`,
    _id: null
  }));

  let uploadedImages = [];
  try {
    const res = await fetch('https://api.ultimatejaipurians.in/upload/images');
    const data = await res.json();
    uploadedImages = Array.isArray(data.images) ? data.images : [];
  } catch (error) {
    console.warn('Could not load uploaded images, showing local images only.', error);
  }

  const allImages = [...localImages, ...uploadedImages];

  allImages.forEach((img) => {
    const card = document.createElement('div');
    card.className = 'production-carousel-card';
    card.dataset.image = img.imageUrl;

    const deleteButton = img._id ? `<span id="${img._id}" class="cross-icon"> x </span>` : '';

    card.innerHTML = `
      <div class="production-in-card">
        ${deleteButton}
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
          initProductionCarousel(); // build AFTER images load
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

  // In case all images are already cached.
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

  const card = e.target.closest('.production-carousel-card');
  if (!card) return;

  const imageUrl = card.dataset.image;
  if (!imageUrl) return;

  openPopPhoto(imageUrl);
});


function openPopPhoto(imageUrl) {
  const popup = document.getElementById('popup-photo');
  const img = document.getElementById('popup-photo-img');
  if (!popup || !img) return;

  img.src = imageUrl;
  popup.style.display = 'flex';

  popup.onclick = (e) => {
    if (e.target.id === 'popup-photo') {
      popup.style.display = 'none';
    }
  };
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

    const activeIndex = getActiveSlideIndex();
    const stageWidth = stage.getBoundingClientRect().width;
    const activeCenterX = (activeIndex * (cardWidth + gap)) + (cardWidth / 2);
    const translateX = activeCenterX - (stageWidth / 2);

    track.style.transform = `translateX(-${translateX}px)`;
    applyCardFocusStates();
  }

  function getActiveSlideIndex() {
    return positionIndex;
  }

  function applyCardFocusStates() {
    if (!slides.length) return;

    const activeIndex = getActiveSlideIndex();

    slides.forEach((slide, idx) => {
      if (idx === activeIndex) {
        slide.classList.add('is-active');
        slide.classList.remove('is-background');
      } else {
        slide.classList.remove('is-active');
        slide.classList.add('is-background');
      }
    });
  }

  function onTransitionEnd() {
    isTransitioning = false;

    if (positionIndex >= visibleCount + slideCount) {
      positionIndex -= slideCount;
      updateTrack(false);
      return;
    }
    if (positionIndex < visibleCount) {
      positionIndex += slideCount;
      updateTrack(false);
      return;
    }

    applyCardFocusStates();
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

  window.addEventListener('resize', () => build());

  build();


}
