// i18n.js is loaded before this script in flipbook.html,
// so window.t() is available synchronously.

const state = {
    currentPageIndex: 0,
    totalPages: 5
};

const elements = {
    flipbookView: document.getElementById('flipbookView'),
    errorMsg:     document.getElementById('errorMsg'),
    prevPage:     document.getElementById('prevPage'),
    nextPage:     document.getElementById('nextPage'),
    pageInfo:     document.getElementById('pageInfo')
};

// Parse URL params
const params       = new URLSearchParams(window.location.search);
const rawImages    = params.get('images') || '';
const imageIndices = rawImages.split(',').map(s => parseInt(s.trim(), 10)).filter(n => !isNaN(n));
const prompt       = decodeURIComponent(params.get('prompt') || '');

// Image slot assignment across 5 pages (9 slots total):
//   Page 0 – Cover   : slot 0
//   Page 1 – Feature : slots 1, 2
//   Page 2 – Story   : slot 3
//   Page 3 – Gallery : slots 4, 5, 6
//   Page 4 – Closing : slots 7, 8
function imgUrl(slot) {
    const idx = imageIndices[slot % imageIndices.length];
    return `/api/image/${idx}`;
}

function buildPageImages(pageIndex) {
    const map = [
        { 'img-1': imgUrl(0) },
        { 'img-1': imgUrl(1), 'img-2': imgUrl(2) },
        { 'img-1': imgUrl(3) },
        { 'img-1': imgUrl(4), 'img-2': imgUrl(5), 'img-3': imgUrl(6) },
        { 'img-1': imgUrl(7), 'img-2': imgUrl(8) },
    ];
    return map[pageIndex];
}

function createLeafletHTML(images, pageIndex) {
    const layouts = ['cover', 'feature', 'story', 'gallery', 'closing'];
    const layout  = layouts[pageIndex];

    if (layout === 'cover') {
        return `
            <div class="leaflet layout-cover">
                <div class="cover-image"><img src="${images['img-1']}" alt="cover"></div>
                <div class="cover-overlay">
                    <span class="cover-tag">${t('leaflet.cover.tag')}</span>
                    <h1 class="cover-title">${t('leaflet.cover.titleLine1')}<br>${t('leaflet.cover.titleLine2')}</h1>
                    <p class="cover-sub">${t('leaflet.cover.sub', { prompt })}</p>
                </div>
            </div>`;
    }

    if (layout === 'feature') {
        return `
            <div class="leaflet layout-feature">
                <div class="feature-main">
                    <div class="feature-img-wrap"><img src="${images['img-1']}" alt="main"></div>
                    <div class="feature-caption">
                        <h2>${t('leaflet.feature.heading')}</h2>
                        <p>${t('leaflet.feature.body')}</p>
                    </div>
                </div>
                <div class="feature-side">
                    <div class="feature-side-img"><img src="${images['img-2']}" alt="side"></div>
                    <div class="feature-side-text">
                        <span class="feature-badge">${t('leaflet.feature.badge')}</span>
                        <h3>${t('leaflet.feature.sideHeading')}</h3>
                        <p>${t('leaflet.feature.sideBody')}</p>
                    </div>
                </div>
            </div>`;
    }

    if (layout === 'story') {
        return `
            <div class="leaflet layout-story">
                <div class="story-photo">
                    <img src="${images['img-1']}" alt="story">
                    <span class="story-photo-label">${t('leaflet.story.photoLabel')}</span>
                </div>
                <div class="story-body">
                    <h2 class="story-heading">${t('leaflet.story.heading')}</h2>
                    <div class="story-columns">
                        <p>${t('leaflet.story.col1')}</p>
                        <p>${t('leaflet.story.col2')}</p>
                    </div>
                </div>
            </div>`;
    }

    if (layout === 'gallery') {
        return `
            <div class="leaflet layout-gallery">
                <div class="gallery-header">
                    <h2>${t('leaflet.gallery.heading')}</h2>
                    <p>${t('leaflet.gallery.sub')}</p>
                </div>
                <div class="gallery-grid">
                    <div class="gallery-item gallery-item-large">
                        <img src="${images['img-1']}" alt="primary">
                        <span class="gallery-label">${t('leaflet.gallery.primaryLabel')}</span>
                    </div>
                    <div class="gallery-item"><img src="${images['img-2']}" alt="detail 1"></div>
                    <div class="gallery-item"><img src="${images['img-3']}" alt="detail 2"></div>
                </div>
            </div>`;
    }

    if (layout === 'closing') {
        return `
            <div class="leaflet layout-closing">
                <div class="closing-stats">
                    <div class="closing-stats-img"><img src="${images['img-1']}" alt="stats"></div>
                    <div class="closing-numbers">
                        <div class="stat-block"><span class="stat-n">${t('leaflet.closing.stat1n')}</span><span class="stat-l">${t('leaflet.closing.stat1l')}</span></div>
                        <div class="stat-block"><span class="stat-n">${t('leaflet.closing.stat2n')}</span><span class="stat-l">${t('leaflet.closing.stat2l')}</span></div>
                    </div>
                </div>
                <div class="closing-cta">
                    <div class="closing-cta-img"><img src="${images['img-2']}" alt="cta"></div>
                    <div class="closing-cta-text">
                        <h2>${t('leaflet.closing.ctaHeading')}</h2>
                        <p>${t('leaflet.closing.ctaBody')}</p>
                    </div>
                </div>
            </div>`;
    }
}

function buildFlipbook(pages) {
    const container = document.getElementById('flipbook');
    container.innerHTML = '';
    pages.forEach((page, index) => {
        const pageEl = document.createElement('div');
        pageEl.className = 'page';
        pageEl.style.zIndex = pages.length - index;

        const content = document.createElement('div');
        content.className = 'page-content';
        content.innerHTML = createLeafletHTML(page.images, index);

        pageEl.appendChild(content);
        container.appendChild(pageEl);
    });
}

function flipBook() {
    document.getElementById('flipbook').querySelectorAll('.page').forEach((page, index) => {
        page.classList.toggle('flipped', index < state.currentPageIndex);
    });
    updateNav();
}

function updateNav() {
    elements.pageInfo.textContent = t('flipbook.pageOf', {
        current: state.currentPageIndex + 1,
        total:   state.totalPages
    });
    elements.prevPage.disabled = state.currentPageIndex === 0;
    elements.nextPage.disabled = state.currentPageIndex === state.totalPages - 1;
}

function init() {
    if (!imageIndices.length) {
        elements.errorMsg.classList.remove('hidden');
        return;
    }

    const pages = Array.from({ length: state.totalPages }, (_, i) => ({
        id: i,
        images: buildPageImages(i)
    }));

    buildFlipbook(pages);
    elements.flipbookView.classList.remove('hidden');
    updateNav();
}

elements.prevPage.addEventListener('click', () => {
    if (state.currentPageIndex > 0) { state.currentPageIndex--; flipBook(); }
});

elements.nextPage.addEventListener('click', () => {
    if (state.currentPageIndex < state.totalPages - 1) { state.currentPageIndex++; flipBook(); }
});

init();
