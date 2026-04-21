// ============================================
// 3D FLIPBOOK COMPARISON
// Two books side-by-side, each with flippable pages
// ============================================

// State
const state = {
    votes: { A: 0, B: 0 },
    pages: [],  // Array of page data
    currentPageIndex: 0,
    totalPages: 0
};

// Elements
const elements = {
    promptInput: document.getElementById('prompt'),
    modelAInput: document.getElementById('modelA'),
    modelBInput: document.getElementById('modelB'),
    numPagesInput: document.getElementById('numPages'),
    generateBtn: document.getElementById('generateBtn'),
    flipbookView: document.getElementById('flipbookView'),
    flipbookA: document.getElementById('flipbookA'),
    flipbookB: document.getElementById('flipbookB'),
    labelA: document.getElementById('labelA'),
    labelB: document.getElementById('labelB'),
    prevPage: document.getElementById('prevPage'),
    nextPage: document.getElementById('nextPage'),
    currentPage: document.getElementById('currentPage'),
    totalPages: document.getElementById('totalPages'),
    countA: document.getElementById('countA'),
    countB: document.getElementById('countB'),
    resetBtn: document.getElementById('resetBtn')
};

// Image slots for each leaflet
const IMAGE_SLOTS = [
    { id: 'img-1', label: 'Hero Image' },
    { id: 'img-2', label: 'Detail 1' },
    { id: 'img-3', label: 'Detail 2' },
    { id: 'img-4', label: 'Feature Image' },
    { id: 'img-5', label: 'Info Image' }
];

// ============================================
// PLACEHOLDER GENERATION
// ============================================

function generatePlaceholder(model, slotId, prompt, pageNum) {
    const canvas = document.createElement('canvas');
    const width = 800;
    const height = 600;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    
    // Color variation based on model and page
    const hue = model === 'A' ? 140 : 200;
    const variation = (parseInt(slotId.split('-')[1]) * 15) + (pageNum * 8);
    
    // Gradient
    const gradient = ctx.createLinearGradient(0, 0, width, height);
    gradient.addColorStop(0, `hsl(${hue + variation}, 45%, 65%)`);
    gradient.addColorStop(1, `hsl(${hue + variation}, 45%, 45%)`);
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, width, height);
    
    // Pattern
    ctx.fillStyle = 'rgba(255, 255, 255, 0.1)';
    for (let i = 0; i < 40; i++) {
        const x = Math.random() * width;
        const y = Math.random() * height;
        const size = Math.random() * 80 + 40;
        ctx.beginPath();
        ctx.arc(x, y, size, 0, Math.PI * 2);
        ctx.fill();
    }
    
    // Text
    ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
    ctx.font = 'bold 28px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(`Model ${model}`, width/2, height/2 - 60);
    
    ctx.font = '22px Arial';
    ctx.fillText(`Page ${pageNum + 1}`, width/2, height/2 - 20);
    
    ctx.font = '18px Arial';
    ctx.fillText(`Slot: ${slotId}`, width/2, height/2 + 20);
    
    ctx.font = 'italic 14px Arial';
    ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
    const shortPrompt = prompt.substring(0, 30);
    ctx.fillText(`"${shortPrompt}..."`, width/2, height/2 + 50);
    
    return canvas.toDataURL('image/png');
}

// ============================================
// GENERATE FLIPBOOKS
// ============================================

async function generateFlipbooks() {
    const prompt = elements.promptInput.value.trim();
    const modelAName = elements.modelAInput.value.trim();
    const modelBName = elements.modelBInput.value.trim();
    const numPages = parseInt(elements.numPagesInput.value) || 5;
    
    if (!prompt) {
        alert('Please enter a prompt');
        return;
    }
    
    // Update labels
    elements.labelA.textContent = modelAName;
    elements.labelB.textContent = modelBName;
    
    // Loading state
    document.body.classList.add('generating');
    elements.generateBtn.disabled = true;
    elements.generateBtn.textContent = 'Generating...';
    
    // Simulate loading
    await new Promise(resolve => setTimeout(resolve, 800));
    
    // Generate pages
    const pages = [];
    for (let i = 0; i < numPages; i++) {
        const page = {
            id: i,
            prompt: prompt,
            modelA: modelAName,
            modelB: modelBName,
            imagesA: {},
            imagesB: {}
        };
        
        // Generate images for this page
        IMAGE_SLOTS.forEach(slot => {
            page.imagesA[slot.id] = generatePlaceholder('A', slot.id, prompt, i);
            page.imagesB[slot.id] = generatePlaceholder('B', slot.id, prompt, i);
        });
        
        pages.push(page);
    }
    
    // Update state
    state.pages = pages;
    state.totalPages = pages.length;
    state.currentPageIndex = 0;
    
    // Build both flipbooks
    buildFlipbook('flipbookA', 'A', pages);
    buildFlipbook('flipbookB', 'B', pages);
    
    // Show flipbook view
    elements.flipbookView.classList.remove('hidden');
    
    // Update UI
    updatePageNavigation();
    
    // Setup vote buttons
    setupVoteButtons();
    
    // Remove loading
    document.body.classList.remove('generating');
    elements.generateBtn.disabled = false;
    elements.generateBtn.textContent = 'Generate New Flipbooks';
    
    // Scroll to view
    elements.flipbookView.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ============================================
// BUILD FLIPBOOK
// ============================================

function buildFlipbook(containerId, model, pages) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    
    pages.forEach((page, index) => {
        const pageEl = document.createElement('div');
        pageEl.className = 'page';
        pageEl.dataset.pageIndex = index;
        
        if (index === 0) {
            pageEl.classList.add('active');
        }
        
        const pageContent = document.createElement('div');
        pageContent.className = 'page-content';
        
        // Create leaflet for this page
        const images = model === 'A' ? page.imagesA : page.imagesB;
        pageContent.innerHTML = createLeafletHTML(images);
        
        pageEl.appendChild(pageContent);
        container.appendChild(pageEl);
    });
}

// ============================================
// CREATE LEAFLET HTML
// ============================================

function createLeafletHTML(images) {
    return `
        <div class="leaflet">
            <!-- Panel 1: Title -->
            <div class="panel panel-title">
                <h1 class="title">Green Spaces<br>Initiative</h1>
                <p class="intro-text">
                    Discover the beauty of nature in your urban environment. 
                    Our parks and gardens provide sanctuary for both wildlife and community.
                </p>
            </div>
            
            <!-- Panel 2: Hero -->
            <div class="panel panel-hero">
                <div class="image-container hero-img">
                    <img src="${images['img-1']}" alt="Hero image">
                </div>
                <div class="overlay-text">
                    <h3>Connecting with Nature</h3>
                    <p>Experience tranquility in the heart of the city</p>
                </div>
            </div>
            
            <!-- Panel 3: Dual -->
            <div class="panel panel-dual">
                <div class="dual-grid">
                    <div class="image-container">
                        <img src="${images['img-2']}" alt="Detail 1">
                    </div>
                    <div class="image-container">
                        <img src="${images['img-3']}" alt="Detail 2">
                    </div>
                </div>
            </div>
            
            <!-- Panel 4: Stats -->
            <div class="panel panel-stats">
                <div class="stats-grid">
                    <div class="stat">
                        <div class="stat-number">15+</div>
                        <div class="stat-label">Parks & Gardens</div>
                    </div>
                    <div class="stat">
                        <div class="stat-number">45%</div>
                        <div class="stat-label">Green Coverage</div>
                    </div>
                    <div class="stat">
                        <div class="stat-number">78%</div>
                        <div class="stat-label">Satisfaction</div>
                    </div>
                </div>
            </div>
            
            <!-- Panel 5: Feature -->
            <div class="panel panel-feature">
                <div class="image-container full-height">
                    <img src="${images['img-4']}" alt="Feature image">
                    <div class="image-caption">
                        Preserving biodiversity for future generations
                    </div>
                </div>
            </div>
            
            <!-- Panel 6: Info -->
            <div class="panel panel-info">
                <h3>Visit Our Spaces</h3>
                <p class="body-text">
                    Open daily from dawn to dusk. Free admission to all public gardens 
                    and park facilities.
                </p>
                <div class="image-container info-img">
                    <img src="${images['img-5']}" alt="Info image">
                </div>
            </div>
        </div>
    `;
}

// ============================================
// PAGE NAVIGATION
// ============================================

function flipToPage(pageIndex) {
    if (pageIndex < 0 || pageIndex >= state.totalPages) return;
    
    const prevIndex = state.currentPageIndex;
    state.currentPageIndex = pageIndex;
    
    // Flip both books simultaneously
    flipBook('flipbookA', prevIndex, pageIndex);
    flipBook('flipbookB', prevIndex, pageIndex);
    
    updatePageNavigation();
}

function flipBook(bookId, fromIndex, toIndex) {
    const book = document.getElementById(bookId);
    const pages = book.querySelectorAll('.page');
    
    if (fromIndex < toIndex) {
        // Flipping forward
        for (let i = fromIndex; i < toIndex; i++) {
            pages[i].classList.add('flipped');
            pages[i].classList.remove('active');
        }
        pages[toIndex].classList.add('active');
    } else {
        // Flipping backward
        for (let i = fromIndex; i > toIndex; i--) {
            pages[i].classList.remove('flipped');
            pages[i].classList.remove('active');
        }
        pages[toIndex].classList.add('active');
        pages[toIndex].classList.remove('flipped');
    }
}

function nextPage() {
    if (state.currentPageIndex < state.totalPages - 1) {
        flipToPage(state.currentPageIndex + 1);
    }
}

function prevPage() {
    if (state.currentPageIndex > 0) {
        flipToPage(state.currentPageIndex - 1);
    }
}

function updatePageNavigation() {
    elements.currentPage.textContent = state.currentPageIndex + 1;
    elements.totalPages.textContent = state.totalPages;
    
    elements.prevPage.disabled = state.currentPageIndex === 0;
    elements.nextPage.disabled = state.currentPageIndex === state.totalPages - 1;
}

// ============================================
// VOTING
// ============================================

function setupVoteButtons() {
    document.querySelectorAll('.btn-vote').forEach(button => {
        button.addEventListener('click', () => {
            const model = button.dataset.model;
            handleVote(model);
        });
    });
}

function handleVote(model) {
    state.votes[model]++;
    updateVoteDisplay();
    
    const button = document.querySelector(`.btn-vote[data-model="${model}"]`);
    button.classList.add('selected');
    setTimeout(() => button.classList.remove('selected'), 800);
    
    console.log(`Vote for Model ${model} on page ${state.currentPageIndex + 1}. Total: A=${state.votes.A}, B=${state.votes.B}`);
}

function updateVoteDisplay() {
    elements.countA.textContent = state.votes.A;
    elements.countB.textContent = state.votes.B;
}

function resetWorkshop() {
    if (confirm('Reset everything?')) {
        state.votes.A = 0;
        state.votes.B = 0;
        state.pages = [];
        state.currentPageIndex = 0;
        state.totalPages = 0;
        
        updateVoteDisplay();
        updatePageNavigation();
        
        elements.flipbookA.innerHTML = '';
        elements.flipbookB.innerHTML = '';
        elements.flipbookView.classList.add('hidden');
        
        elements.generateBtn.textContent = 'Generate Flipbooks';
        
        console.log('Workshop reset');
    }
}

// ============================================
// EVENT LISTENERS
// ============================================

elements.generateBtn.addEventListener('click', generateFlipbooks);
elements.promptInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') generateFlipbooks();
});

elements.prevPage.addEventListener('click', prevPage);
elements.nextPage.addEventListener('click', nextPage);
elements.resetBtn.addEventListener('click', resetWorkshop);

// Keyboard navigation
document.addEventListener('keydown', (e) => {
    if (elements.flipbookView.classList.contains('hidden')) return;
    
    if (e.key === 'ArrowLeft') prevPage();
    if (e.key === 'ArrowRight') nextPage();
});

// ============================================
// INITIALIZATION
// ============================================

console.log('3D Flipbook Comparison - Ready');
console.log('Generate flipbooks to start. Use arrow keys to flip pages!');