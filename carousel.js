document.addEventListener('DOMContentLoaded', async () => {
  console.log('it is working....');

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

    const deleteButton = img._id ? `<span id="${img._id}" class="cross-icon"> × </span>` : '';

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

  // Track loading status
  const checkAllLoaded = () => {
    loaded++;
    if (loaded === images.length) {
      initProductionCarousel();
    }
  };

  images.forEach(img => {
    if (img.complete) {
      checkAllLoaded();
    } else {
      img.onload = checkAllLoaded;
      img.onerror = checkAllLoaded;
    }
  });

  // In case all images are already cached or there are no images
  if (images.length === 0) {
    initProductionCarousel();
  }
});

// Moved removePhoto outside DOMContentLoaded but still in global scope
async function removePhoto(imgId, key) {
  console.log('api called...');
  if (key === "Reunion") {
    try {
      const res = await fetch(`https://api.ultimatejaipurians.in/upload/delete?id=${imgId}`, {
        method: "DELETE",
        headers: { 'Content-Type': 'application/json' },
      });

      const data = await res.json();
      const { status } = data;

      if (!status) {
        alert('Photo is not deleted. Try again later.');
      } else {
        alert('Photo is deleted.');
        window.location.reload();
      }
    } catch (error) {
      console.error('Delete failed:', error);
      alert('Failed to delete photo. Please try again.');
    }
  } else {
    alert('Your Pass Key is wrong');
  }
}

// Event listener setup - wrapped in a function to ensure it's called after DOM is ready
function setupEventListeners() {
  const track = document.getElementById('track-pro');
  if (!track) return;

  // Remove existing listener to prevent duplicates
  track.removeEventListener('click', handleTrackClick);
  track.addEventListener('click', handleTrackClick);
}

// Separate handler function for better organization
function handleTrackClick(e) {
  // X button clicked logic
  if (e.target.classList.contains('cross-icon')) {
    e.stopPropagation(); // Prevent card click from firing
    
    const imageId = e.target.id;
    console.log('X clicked, Image ID:', imageId);

    const key = window.prompt('Enter your pass key:');
    if (key !== null) { // Only proceed if user didn't cancel
      removePhoto(imageId, key);
    }
    return;
  }

  const card = e.target.closest('.production-carousel-card');
  if (!card) return;

  const imageUrl = card.dataset.image;
  if (!imageUrl) return;

  openPopPhoto(imageUrl);
}

function openPopPhoto(imageUrl) {
  const popup = document.getElementById('popup-photo');
  const img = document.getElementById('popup-photo-img');
  const closeButton = document.getElementById('popup-photo-close');
  
  if (!popup || !img) return;

  img.src = imageUrl;
  popup.style.display = 'flex';

  // Clean up previous listeners
  const newCloseButton = closeButton.cloneNode(true);
  closeButton.parentNode.replaceChild(newCloseButton, closeButton);
  
  newCloseButton.onclick = () => {
    popup.style.display = 'none';
  };

  // Use once option for popup background click
  const handlePopupClick = (e) => {
    if (e.target === popup) {
      popup.style.display = 'none';
      popup.removeEventListener('click', handlePopupClick);
    }
  };
  
  popup.removeEventListener('click', handlePopupClick);
  popup.addEventListener('click', handlePopupClick);
}

function initProductionCarousel() {
  const track = document.getElementById('track-pro');
  const stage = document.getElementById('stage-pro');
  const prevBtn = document.getElementById('prev-pro');
  const nextBtn = document.getElementById('next-pro');

  // Check if elements exist
  if (!track || !stage || !prevBtn || !nextBtn) {
    console.error('Required carousel elements not found');
    return;
  }

  const baseSlides = Array.from(track.querySelectorAll('.production-carousel-card')).map((card) => {
    const cleanCard = card.cloneNode(true);
    cleanCard.classList.remove('clone', 'is-active', 'is-background');
    return cleanCard;
  });

  let slideCount = baseSlides.length;
  let visibleCount = getVisibleCount();
  let slides = [];
  let positionIndex = 0;
  const transitionMs = 450;
  let isTransitioning = false;
  let cardWidth = 0;
  let gap = 0;
  let resizeTimer = null;

  function getVisibleCount() {
    if (window.matchMedia('(max-width:520px)').matches) return 1;
    if (window.matchMedia('(max-width:900px)').matches) return 2;
    return 3;
  }

  function build() {
    if (!slideCount) {
      console.warn('Carousel build skipped: no slides');
      return;
    }

    track.style.transition = 'none';
    track.innerHTML = '';

    visibleCount = getVisibleCount();

    const originals = baseSlides.map((card) => {
      const clone = card.cloneNode(true);
      clone.classList.remove('clone', 'is-active', 'is-background');
      return clone;
    });

    const frontClones = [];
    const endClones = [];

    for (let i = 0; i < visibleCount; i++) {
      const idxFront = (slideCount - visibleCount + i + slideCount) % slideCount;
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

    // Re-attach event listeners after rebuilding
    setupEventListeners();
  }

  function computeCardDimensions() {
    const card = track.querySelector('.production-carousel-card:not(.clone)') || track.querySelector('.production-carousel-card');
    if (!card) return;

    // offsetWidth ignores transform scaling, which keeps centering math stable.
    cardWidth = card.offsetWidth;
    gap = parseFloat(getComputedStyle(track).gap) || 30;
  }

  function updateTrack(animate = true) {
    track.style.transition = animate
      ? `transform ${transitionMs}ms cubic-bezier(.25,.8,.25,1)`
      : 'none';

    const activeIndex = getActiveSlideIndex();
    const stageWidth = stage.getBoundingClientRect().width;
    if (!cardWidth || !stageWidth) {
      computeCardDimensions();
    }
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
      slide.classList.toggle('is-active', idx === activeIndex);
      slide.classList.toggle('is-background', idx !== activeIndex);
    });
  }

  function onTransitionEnd(e) {
    if (e.target !== track || e.propertyName !== 'transform') {
      return;
    }

    isTransitioning = false;

    if (positionIndex >= visibleCount + slideCount) {
      positionIndex -= slideCount;
      updateTrack(false);
    } else if (positionIndex < visibleCount) {
      positionIndex += slideCount;
      updateTrack(false);
    }

    applyCardFocusStates();
  }

  function next() {
    if (isTransitioning || !slideCount) return;
    isTransitioning = true;
    positionIndex++;
    updateTrack(true);
  }

  function prev() {
    if (isTransitioning || !slideCount) return;
    isTransitioning = true;
    positionIndex--;
    updateTrack(true);
  }

  // Handle resize with debouncing
  function handleResize() {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      build();
    }, 150);
  }

  function handleHashChange() {
    computeCardDimensions();
    updateTrack(false);
  }

  // Clean up old listeners
  track.removeEventListener('transitionend', onTransitionEnd);
  prevBtn.removeEventListener('click', prev);
  nextBtn.removeEventListener('click', next);
  window.removeEventListener('resize', handleResize);
  window.removeEventListener('hashchange', handleHashChange);

  // Add new listeners
  track.addEventListener('transitionend', onTransitionEnd);
  prevBtn.addEventListener('click', prev);
  nextBtn.addEventListener('click', next);
  window.addEventListener('resize', handleResize);
  window.addEventListener('hashchange', handleHashChange);

  // Initial build
  build();
  setupEventListeners();
}