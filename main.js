const BACKEND_URL_OF_INDEX = "https://api.ultimatejaipurians.in";
// const KEY_ID = "rzp_test_S7GF05YfeBu3dp";
// const BACKEND_URL_OF_INDEX = "http://localhost:8000";

//-------------------------------
//       Venue Cards PopUp
//-------------------------------

const overlay = document.getElementById("eventPopupOverlay");
const closeBtn = document.getElementById("eventPopupClose");

// Popup text elements
const titleEl = document.getElementById("popupTitle");
const dateEl = document.getElementById("popupDate");
const timeEl = document.getElementById("popupTime");
const address = document.getElementById("event-address");
const popupIcon = document.getElementById("popupIcon");
const dress = document.getElementById("dress-code");

// All cards
const cards = document.querySelectorAll(".eventPopupTrigger");
const openMap = document.getElementById("openMap");

// When clicking any card
cards.forEach((card) => {
  card.addEventListener("click", () => {
    titleEl.textContent = card.dataset.title;
    dateEl.textContent = card.dataset.date;
    popIcon.setAttribute("src", card.dataset.icon);
    dress.textContent = card.dataset.dress;
    address.textContent = card.dataset.address;

    // popupIcon.innerHTML = `<img id="popupIcon" src="${card.dataset.icon}" style="width:44px;height:44px;"></img>`;

    const title = titleEl.textContent;

    if (title === "Dinner") {
      openMap.style.display = "flex";
      openMap.textContent = "Open To Map";
      openMap.setAttribute(
        "href",
        "https://maps.app.goo.gl/DSnifjemDkRE39xM9?g_st=aw",
      );
    } else if (title === "Lunch") {
      openMap.style.display = "flex";
      openMap.textContent = "Open Map";
      openMap.setAttribute(
        "href",
        "https://maps.app.goo.gl/u2v1BB323MhPc8GS6?g_st=aw",
      );
    }

    // re-render lucide icons
    requestAnimationFrame(() => lucide.createIcons());

    overlay.style.display = "flex";
  });
});

const popIcon = document.getElementById("popupIcon");

// Close popup
closeBtn.addEventListener("click", () => {
  overlay.style.display = "none";
  openMap.style.display = "none";
  popIcon.setAttribute("src", "data-icon");
});

// Click outside to close
overlay.addEventListener("click", (e) => {
  if (e.target === overlay) {
    overlay.style.display = "none";
    openMap.style.display = "none";
  }
});

//------------------------------------
//     Photo Pop Up Open Logic
//------------------------------------

function openPopPhoto(imgSrc) {
  document.getElementById("popup-photo-img").src = imgSrc;
  document.getElementById("popup-photo").style.display = "flex";

  function closePopup() {
    document.getElementById("popup-photo").style.display = "none";
  }

  document.getElementById("popup-photo").addEventListener("click", (e) => {
    if (e.target.id === "popup-photo") closePopup();
  });
}

// ---------------------------
//           Loader
//--------------------------
const button = document.getElementById("downloadBtn");

button.addEventListener("click", () => {
  // show loader
  button.classList.add("loading");
});

//----------------------------------------------
//      Change No. of People Compilation
//----------------------------------------------

const input = document.querySelectorAll(".someone-check");

input.forEach((el) => {
  el.addEventListener("change", function () {
    if (this.checked) {
      let selectedValue = this.value;

      const peopleChange = document.getElementById("attend");

      if (selectedValue === "stag") {
        peopleChange.innerText = 1;
      } else {
        peopleChange.innerText = 2;
      }
    }
  });
});

//----------------------------------------
//   Toast Pop up
//----------------------------------------
const infoBtn = document.getElementById("info");
const panToast = document.getElementById("panToast");

let pantoastTimer = null;

infoBtn.addEventListener("click", () => {
  // Show the toast
  panToast.classList.add("show");

  if (pantoastTimer) clearTimeout(pantoastTimer);

  pantoastTimer = setTimeout(() => {
    panToast.classList.remove("show");
  }, 5000);
});

//----------------------------------------
//       Donation Pop Up open
//----------------------------------------

const donationCheckboxes = document.querySelectorAll(".donation-check");
const donationtoast = document.getElementById("toast");
const toastAmount = document.getElementById("toastAmount");

let toastTimer = null;

function showToast(event) {
  const checkbox = event.target;

  // show only when checked
  if (!checkbox.checked) return;

  toastAmount.textContent = checkbox.value;
  donationtoast.classList.add("show");

  // reset timer if triggered again
  if (toastTimer) clearTimeout(toastTimer);

  toastTimer = setTimeout(() => {
    donationtoast.classList.remove("show");
  }, 3000);
}

donationCheckboxes.forEach((cb) => {
  cb.addEventListener("change", showToast);
});

//----------------------------------------------
//             Grand Total
//----------------------------------------------

document.addEventListener("DOMContentLoaded", () => {
  // ===== DEADLINE CHECK =====
  const registrationDeadline = new Date('2026-03-08T23:59:59'); // March 8, 2026
  const now = new Date();

  if (now > registrationDeadline) {
    const registerBtns = document.querySelectorAll('button[id="openPopup"], .btn-primary[href="#register"], .mobile-link[href="#register"], #openPopup');
    registerBtns.forEach(btn => {
      if (btn.tagName === 'BUTTON' || btn.classList.contains('btn-submit')) {
        btn.disabled = true;
        btn.textContent = "REGISTRATION CLOSED";
        btn.style.background = "#333";
        btn.style.color = "#666";
        btn.style.cursor = "not-allowed";
        btn.style.boxShadow = "none";
        btn.style.opacity = "0.7";
      } else {
        btn.style.pointerEvents = "none";
        btn.style.opacity = "0.4";
        btn.textContent = "Closed";
      }
    });

    const contactForm = document.getElementById('contactForm');
    if (contactForm) {
      const allFormElements = contactForm.querySelectorAll('input, textarea, select, checkbox, .btn-submit');
      allFormElements.forEach(el => {
        el.disabled = true;
        el.style.opacity = "0.6";
      });
      const deadlineMsg = document.createElement('div');
      deadlineMsg.style.color = "var(--accent-pink)";
      deadlineMsg.style.textAlign = "center";
      deadlineMsg.style.marginTop = "20px";
      deadlineMsg.style.fontWeight = "bold";
      deadlineMsg.textContent = "Registration is now closed.";
      contactForm.prepend(deadlineMsg);
    }
  }

  // ===== ELEMENTS =====
  const attendSpan = document.getElementById("attend");
  const grandTotalEl = document.getElementById("grandTotal");

  const stagCheckbox = document.querySelector('.someone-check[value="stag"]');
  const plusOneCheckbox = document.querySelector(
    '.someone-check[value="noStag"]',
  );

  const donationCheckboxes = document.querySelectorAll(".donation-check");

  // ===== SAFETY CHECK =====
  if (!attendSpan || !grandTotalEl || !stagCheckbox || !plusOneCheckbox) {
    console.error("Required elements not found in DOM");
    return;
  }

  // ===== ATTENDEES COUNT =====
  function getAttendeesCount() {
    const stag = stagCheckbox.checked;
    const plusOne = plusOneCheckbox.checked;

    // If +1 is checked but stag is not → assume primary also coming
    if (plusOne && !stag) return 2;

    return (stag ? 1 : 0) + (plusOne ? 1 : 0);
  }

  // ===== PASS PRICE RULES =====
  function getPassPrice(count) {
    if (count === 1) return 5000;
    if (count === 2) return 7500;
    return 0;
  }

  // ===== DONATION TOTAL =====
  function getDonationTotal() {
    let total = 0;

    donationCheckboxes.forEach((cb) => {
      if (cb.checked) {
        total += Number(cb.value);
      }
    });

    return total;
  }

  // ===== MAIN CALCULATION =====
  function calculateAndRender() {
    const attendees = getAttendeesCount();
    attendSpan.textContent = attendees;

    const passValue = getPassPrice(attendees);
    const donationValue = getDonationTotal();

    const grandTotal = passValue + donationValue;

    grandTotalEl.textContent = grandTotal.toLocaleString("en-IN");
  }

  // ===== EVENTS =====
  stagCheckbox.addEventListener("change", calculateAndRender);
  plusOneCheckbox.addEventListener("change", calculateAndRender);

  donationCheckboxes.forEach((cb) => {
    cb.addEventListener("change", calculateAndRender);
  });

  // ===== INITIAL RENDER =====
  calculateAndRender();
});

//=======================================================================
//=======================================================================

//----------------------------------------------
//             Create Paytm Order here
//----------------------------------------------

async function paymentRazorpay() {
  const name = document.getElementById("name").value.trim();
  const phone = Number(document.getElementById("number").value.trim());
  const email = document.getElementById("email").value.trim();
  const address = document.getElementById("address").value.trim();
  const city = document.getElementById("city").value.trim();
  const adhaar = document.getElementById("adhaar").value.trim();

  const noPeople = Number(document.getElementById("attend").textContent);
  const donationCheckboxes = document.querySelectorAll(".donation-check");

  const someone = document.querySelector(".someone-check:checked")?.value || "";
  const support = document.querySelector(".support-check:checked")?.value || "";
  const size = document.querySelector(".size-check:checked")?.value || "";
  const section10 =
    document.querySelector(".section-10-check:checked")?.value || "";
  const section12 =
    document.querySelector(".section-12-check:checked")?.value || "";

  if (!size) {
    Toastify({
      text: "Please select your preferred jacket size",
      duration: 3000,
      gravity: "top",
      position: "center",
      style: {
        background: "linear-gradient(to right, #ff5f6d, #ffc371)",
      },
    }).showToast();
    return;
  }

  const donateAmt = getDonationTotal();
  function getDonationTotal() {
    let total = 0;

    donationCheckboxes.forEach((cb) => {
      if (cb.checked) {
        total += Number(cb.value);
      }
    });

    return total;
  }

  let passValue;
  if (noPeople == 1) {
    passValue = 5000;
  } else if (noPeople == 2) {
    passValue = 7500;
  } else {
    passValue = 0;
  }

  const passDonateValue = passValue + donateAmt;
  // + donateAmt;
  const payValue = (5 * passDonateValue) / 100;
  const totalPayPrice = passDonateValue + payValue;

  const formData = {
    name,
    phone,
    email,
    address,
    city,
    adhaar,
    noPeople,
    donateAmt,
    someone,
    support,
    size,
    section10,
    section12,
    passValue,
    totalPayPrice,
  };

  console.log(formData);
  // we are handing out jackets as souvenirs, please tell us your preferred size. 
  // S, M, L, XL, XXL. 
  // console.log(totalPayPrice);

  async function initiatePayment() {
    try {
      const response = await fetch(`${BACKEND_URL_OF_INDEX}/paytm/create-order`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          amount: totalPayPrice,
          formData: formData,
        }),
      });

      const data = await response.json();
      // console.log("Response from server:", data);

      if (!response.ok) {
        // alert(data.message || "Something went wrong");
        // document.getElementById("downloadBtn").classList.remove("loading");
        return;
      }

      //----- use paytm docs code ----------------//

      async function onScriptLoad() {
        var config = {
          root: "",
          flow: "DEFAULT",
          data: {
            orderId: data.orderId,
            token: data.txnToken,
            tokenType: "TXN_TOKEN",
            amount: data.amount,
          },
          handler: {
            notifyMerchant: function (eventName, data) {
              console.log("notifyMerchant handler function called");
              console.log("eventName => ", eventName);
              console.log("data => ", data);
              if ("App closed from the header icon" == data.message) {
                document
                  .getElementById("downloadBtn")
                  .classList.remove("loading");
              }
            },
          },
        };

        if (window.Paytm && window.Paytm.CheckoutJS) {
          console.log("Paytm checkout window runing start");

          await window.Paytm.CheckoutJS.init(config);
          window.Paytm.CheckoutJS.invoke();
        }
      }

      onScriptLoad();
    } catch (error) {
      console.error("Payment initiation error:", error);
      alert("Error connecting to server. Please check your internet or try again.");
      document.getElementById("downloadBtn").classList.remove("loading");
    }
  }

  initiatePayment();
}

//======================================================================

// async function testingCreateOrder() {
//   const name = document.getElementById("name").value.trim();
//   const phone = Number(document.getElementById("number").value.trim());
//   const email = document.getElementById("email").value.trim();
//   const address = document.getElementById("address").value.trim();
//   const city = document.getElementById("city").value.trim();
//   const adhaar = document.getElementById("adhaar").value.trim();

//   const noPeople = Number(document.getElementById("attend").textContent);
//   // let donateAmt = Number(document.getElementById("donate").value.trim());
//   const donationCheckboxes = document.querySelectorAll(".donation-check");

//   const someone = document.querySelector(".someone-check:checked")?.value || "";
//   const support = document.querySelector(".support-check:checked")?.value || "";
//   const size = document.querySelector(".size-check:checked")?.value || "";
//   const section10 =
//     document.querySelector(".section-10-check:checked")?.value || "";
//   const section12 =
//     document.querySelector(".section-12-check:checked")?.value || "";

//   const donateAmt = getDonationTotal();
//   function getDonationTotal() {
//     let total = 0;

//     donationCheckboxes.forEach((cb) => {
//       if (cb.checked) {
//         total += Number(cb.value);
//       }
//     });

//     return total;
//   }

//   let passValue;
//   if (noPeople == 1) {
//     passValue = 5000;
//   } else if (noPeople == 2) {
//     passValue = 7500;
//   } else {
//     passValue = 0;
//   }

//   const passDonateValue = passValue + donateAmt;
//   // + donateAmt;
//   const payValue = (7 * passDonateValue) / 100;
//   const totalPayPrice = passDonateValue + payValue;

//   const formData = {
//     name,
//     phone,
//     email,
//     address,
//     city,
//     adhaar,
//     noPeople,
//     donateAmt,
//     someone,
//     support,
//     size,
//     section10,
//     section12,
//     passValue,
//     totalPayPrice,
//   };

//   async function initiatePayment() {
//     const response = await fetch(`${BACKEND_URL_OF_INDEX}/paytm/create-order`, {
//       method: "POST",
//       headers: { "Content-Type": "application/json" },
//       body: JSON.stringify({
//         amount: 1, //totalPayPrice final production time will open
//         formData: formData,
//       }),
//     });

//     const data = await response.json();

//     //----- use paytm docs code ----------------//

//     async function onScriptLoad() {
//       var config = {
//         root: "",
//         flow: "DEFAULT",
//         data: {
//           orderId: data.orderId /* update order id */,
//           token: data.txnToken /* update token value */,
//           tokenType: "TXN_TOKEN",
//           amount: data.amount /* update amount */,
//         },
//         handler: {
//           notifyMerchant: function (eventName, data) {
//             console.log("notifyMerchant handler function called");
//             console.log("eventName => ", eventName);
//             console.log("data => ", data);
//             if ("App closed from the header icon" == data.message) {
//               document
//                 .getElementById("downloadBtn")
//                 .classList.remove("loading");
//             }
//           },
//         },
//       };

//       if (window.Paytm && window.Paytm.CheckoutJS) {
//         console.log("Paytm checkout window runing start");

//         await window.Paytm.CheckoutJS.init(config);
//         window.Paytm.CheckoutJS.invoke();
//       }
//     }

//     onScriptLoad();
//   }

//   initiatePayment();
// }

// Allow only one checkbox from a group
function allowOnlyOne(className) {
  const boxes = document.querySelectorAll("." + className);
  boxes.forEach((box) => {
    box.addEventListener("change", () => {
      if (box.checked) {
        boxes.forEach((other) => {
          if (other !== box) other.checked = false;
        });
      }
    });
  });
}

// allowOnlyOne("someone-check");
allowOnlyOne("support-check");
allowOnlyOne("size-check");
allowOnlyOne("section-10-check");
allowOnlyOne("section-12-check");

// Mobile Menu controls
document.addEventListener("DOMContentLoaded", () => {
  lucide.createIcons();
});

const menuBtn = document.getElementById("menu-btn");
const mobileMenu = document.getElementById("mobile-menu");
const closeMenuBtn = document.getElementById("close-menu");
const mobileLinks = document.querySelectorAll(".mobile-link");

function toggleMenu() {
  const open = mobileMenu.classList.contains("open");
  if (!open) {
    mobileMenu.classList.add("open");
    mobileMenu.setAttribute("aria-hidden", "false");
    document.body.style.overflow = "hidden";
  } else {
    mobileMenu.classList.remove("open");
    mobileMenu.setAttribute("aria-hidden", "true");
    document.body.style.overflow = "";
  }
}

menuBtn.addEventListener("click", toggleMenu);
closeMenuBtn.addEventListener("click", toggleMenu);
mobileLinks.forEach((link) => link.addEventListener("click", toggleMenu));

// Form submission simulation + toast
const form = document.getElementById("contactForm");
const toast = document.getElementById("toast");

//COntact form whatsapp send

document.getElementById("contact-btn").addEventListener("click", function (e) {
  e.preventDefault();

  const name = document.getElementById("contact-name").value;
  const phone = document.getElementById("contact-phone").value;
  const email = document.getElementById("contact-email").value;
  const message = document.getElementById("contact-text").value;

  const whatsappNumber = "916388505758"; // with country code

  const text = `
Name: ${name}
Phone: ${phone}
email: ${email}
message: ${message}
  `;

  const url = `https://wa.me/${whatsappNumber}?text=${encodeURIComponent(
    text,
  )}`;

  window.open(url, "_blank");
});

// Stop Music Logic

const music = document.getElementById("music");
const noMusic = document.getElementsByClassName("stop-music");
const audio = document.getElementById("bgMusic");

audio.volume = 0.01;
let checkMusic;
music.addEventListener("click", () => {
  if (!checkMusic) {
    noMusic[0].style.display = "block";
    audio.pause();
    checkMusic = true;
  } else {
    noMusic[0].style.display = "none";
    audio.play();
    checkMusic = false;
  }
});
