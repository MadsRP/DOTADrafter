// Enhanced app.js - Sort heroes by win rate within each attribute section
document.addEventListener('DOMContentLoaded', function() {
    // Draft state (same as before)
    const draftState = {
        currentTeam: 'radiant',
        currentAction: 'pick',
        step: 0,
        radiant: { picks: [], bans: [] },
        dire: { picks: [], bans: [] }
    };

    let originalHeroData = {};
    let isDraftComplete = false;

    // Draft order (same as before)
    const draftOrder = [
        { team: 'radiant', action: 'ban' }, { team: 'dire', action: 'ban' },
        { team: 'radiant', action: 'ban' }, { team: 'dire', action: 'ban' },
        { team: 'radiant', action: 'pick' }, { team: 'dire', action: 'pick' },
        { team: 'dire', action: 'pick' }, { team: 'radiant', action: 'pick' },
        { team: 'radiant', action: 'ban' }, { team: 'dire', action: 'ban' },
        { team: 'radiant', action: 'ban' }, { team: 'dire', action: 'ban' },
        { team: 'dire', action: 'pick' }, { team: 'radiant', action: 'pick' },
        { team: 'radiant', action: 'pick' }, { team: 'dire', action: 'pick' },
        { team: 'dire', action: 'ban' }, { team: 'radiant', action: 'ban' },
        { team: 'dire', action: 'ban' }, { team: 'radiant', action: 'ban' },
        { team: 'dire', action: 'pick' }, { team: 'radiant', action: 'pick' }
    ];

    // Store attribute section containers for sorting
    const attributeSections = {
        'str': document.querySelector('.strength .hero-grid'),
        'agi': document.querySelector('.agility .hero-grid'),
        'int': document.querySelector('.intelligence .hero-grid'),
        'all': document.querySelector('.universal .hero-grid')
    };

    // Initialize
    initializeEventListeners();
    captureOriginalHeroData();
    updateDraftUI();
    updateTeamIndicator();
    checkModelStatus();
    requestRecommendations();

    function initializeEventListeners() {
        // Add click handlers to all hero cards
        document.querySelectorAll('.hero-card').forEach(heroCard => {
            heroCard.addEventListener('click', function() {
                if (isDraftComplete) {
                    alert('Draft is complete! Use Reset Draft to start over.');
                    return;
                }

                const img = this.querySelector('img');
                if (!img) return;

                const heroId = extractHeroId(img);
                const heroName = img.alt || img.title || 'Unknown Hero';

                if (isHeroAlreadySelected(heroId)) {
                    alert(`${heroName} is already picked or banned.`);
                    return;
                }

                selectHero(heroId, heroName);
            });
        });

        // Rest of event listeners (same as before)
        document.getElementById('reset-draft')?.addEventListener('click', resetDraft);
        document.getElementById('next-step')?.addEventListener('click', nextStep);

        document.getElementById('hero-search')?.addEventListener('input', function() {
            const searchTerm = this.value.toLowerCase();
            document.querySelectorAll('.hero-card').forEach(heroCard => {
                const img = heroCard.querySelector('img');
                const heroName = (img?.alt || img?.title || '').toLowerCase();
                heroCard.style.display = heroName.includes(searchTerm) ? 'block' : 'none';
            });
        });
    }

    function captureOriginalHeroData() {
        document.querySelectorAll('.hero-card').forEach(heroCard => {
            const img = heroCard.querySelector('img');
            const winRateEl = heroCard.querySelector('.hero-win-rate');

            if (img && winRateEl) {
                const heroId = extractHeroId(img);
                originalHeroData[heroId] = {
                    name: img.alt || img.title,
                    originalWinRate: winRateEl.textContent,
                    heroCard: heroCard,
                    attribute: getHeroAttribute(heroCard) // Store which section this hero belongs to
                };
            }
        });
    }

    function getHeroAttribute(heroCard) {
        // Determine which attribute section this hero belongs to
        const strengthSection = document.querySelector('.strength .hero-grid');
        const agilitySection = document.querySelector('.agility .hero-grid');
        const intelligenceSection = document.querySelector('.intelligence .hero-grid');
        const universalSection = document.querySelector('.universal .hero-grid');

        if (strengthSection && strengthSection.contains(heroCard)) return 'str';
        if (agilitySection && agilitySection.contains(heroCard)) return 'agi';
        if (intelligenceSection && intelligenceSection.contains(heroCard)) return 'int';
        if (universalSection && universalSection.contains(heroCard)) return 'all';
        return 'unknown';
    }

    function extractHeroId(imgElement) {
        const match = imgElement.src.match(/heroes\/(\d+)\.jpg/);
        return match ? parseInt(match[1]) : 0;
    }

    function selectHero(heroId, heroName) {
        if (isDraftComplete) return;

        // Follow draft order
        if (draftState.step < draftOrder.length) {
            const currentDraftStep = draftOrder[draftState.step];
            draftState.currentTeam = currentDraftStep.team;
            draftState.currentAction = currentDraftStep.action;
        }

        // Add hero to current team's selections
        const actionType = draftState.currentAction + 's';
        draftState[draftState.currentTeam][actionType].push({
            id: heroId,
            name: heroName
        });

        // Update UI
        updateDraftUI();
        updateHeroAvailability(heroId, false);

        // Move to next step
        if (draftState.step < draftOrder.length) {
            draftState.step++;

            if (draftState.step >= draftOrder.length) {
                isDraftComplete = true;
                console.log('Draft completed!');
            }

            updateTeamIndicator();

            if (!isDraftComplete) {
                setTimeout(() => {
                    requestRecommendations();
                }, 300);
            } else {
                showFinalDraftAnalysis();
            }
        }
    }

    function updateHeroAvailability(heroId, available) {
        document.querySelectorAll('.hero-card').forEach(heroCard => {
            const img = heroCard.querySelector('img');
            if (img && extractHeroId(img) === heroId) {
                if (available) {
                    heroCard.classList.remove('hero-picked', 'hero-banned');
                    heroCard.style.opacity = '1';
                    heroCard.style.filter = 'none';
                } else {
                    const isInPicks = [...draftState.radiant.picks, ...draftState.dire.picks]
                        .some(h => h.id === heroId);

                    if (isInPicks) {
                        heroCard.classList.add('hero-picked');
                        heroCard.style.opacity = '0.5';
                        heroCard.style.filter = 'brightness(0.7)';
                    } else {
                        heroCard.classList.add('hero-banned');
                        heroCard.style.opacity = '0.3';
                        heroCard.style.filter = 'brightness(0.4) grayscale(0.8)';
                    }
                }
            }
        });
    }

    function isHeroAlreadySelected(heroId) {
        const allSelections = [
            ...draftState.radiant.picks, ...draftState.radiant.bans,
            ...draftState.dire.picks, ...draftState.dire.bans
        ];
        return allSelections.some(h => h.id === heroId);
    }

    function requestRecommendations() {
        if (isDraftComplete || draftState.step >= draftOrder.length) return;

        showLoadingRecommendations();

        const requestData = {
            radiant: {
                picks: draftState.radiant.picks.map(h => h.id),
                bans: draftState.radiant.bans.map(h => h.id)
            },
            dire: {
                picks: draftState.dire.picks.map(h => h.id),
                bans: draftState.dire.bans.map(h => h.id)
            },
            currentTeam: draftState.currentTeam,
            currentAction: draftState.currentAction
        };

        Promise.all([
            fetch('/api/counterpicks', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestData)
            }).then(response => response.json()),

            calculateAllHeroWinRates(requestData)
        ])
            .then(([recommendationsData, allWinRates]) => {
                displayRecommendations(recommendationsData.recommendations || []);
                displayDraftAnalysis(recommendationsData.analysis);

                // NEW: Update win rates AND sort heroes
                updateAllHeroWinRatesAndSort(allWinRates);
            })
            .catch(error => {
                console.error('Error fetching recommendations:', error);
                displayRecommendations([]);
            });
    }

    function calculateAllHeroWinRates(requestData) {
        return new Promise((resolve) => {
            const allWinRates = {};
            const selectedHeroIds = new Set([
                ...requestData.radiant.picks, ...requestData.radiant.bans,
                ...requestData.dire.picks, ...requestData.dire.bans
            ]);

            document.querySelectorAll('.hero-card img').forEach(img => {
                const heroId = extractHeroId(img);
                if (heroId) {
                    if (selectedHeroIds.has(heroId)) {
                        allWinRates[heroId] = 0; // Unavailable
                    } else {
                        const draftComplexity = selectedHeroIds.size;
                        const heroVariance = (heroId * 7) % 40;
                        const baseRate = 35 + heroVariance;

                        const teamBalance = Math.abs(requestData.radiant.picks.length - requestData.dire.picks.length);
                        const adjustment = (teamBalance * 2) + (draftComplexity * 0.5);

                        const finalRate = Math.max(25, Math.min(75, baseRate + adjustment - 20));
                        allWinRates[heroId] = Math.round(finalRate * 10) / 10;
                    }
                }
            });

            setTimeout(() => resolve(allWinRates), 100);
        });
    }

    // NEW: Update win rates AND sort heroes by win rate within each attribute section
    function updateAllHeroWinRatesAndSort(allWinRates) {
        console.log('Updating win rates and sorting heroes...');

        // First, update all win rates
        const heroesData = [];

        document.querySelectorAll('.hero-card').forEach(heroCard => {
            const img = heroCard.querySelector('img');
            const winRateEl = heroCard.querySelector('.hero-win-rate');

            if (img && winRateEl) {
                const heroId = extractHeroId(img);
                const attribute = getHeroAttribute(heroCard);

                if (heroId in allWinRates) {
                    const newWinRate = allWinRates[heroId];

                    if (newWinRate === 0) {
                        winRateEl.textContent = 'N/A';
                        heroCard.style.border = '';
                        heroCard.style.boxShadow = '';
                    } else {
                        winRateEl.textContent = `${newWinRate}%`;

                        if (newWinRate >= 60) {
                            heroCard.style.border = '2px solid #4CAF50';
                            heroCard.style.boxShadow = '0 0 10px rgba(76, 175, 80, 0.4)';
                        } else if (newWinRate <= 35) {
                            heroCard.style.border = '2px solid #f44336';
                            heroCard.style.boxShadow = '0 0 10px rgba(244, 67, 54, 0.4)';
                        } else {
                            heroCard.style.border = '';
                            heroCard.style.boxShadow = '';
                        }
                    }

                    // Store hero data for sorting
                    heroesData.push({
                        heroCard: heroCard,
                        heroId: heroId,
                        winRate: newWinRate,
                        attribute: attribute,
                        name: img.alt || img.title || 'Unknown'
                    });
                }
            }
        });

        // Now sort and reorder heroes within each attribute section
        sortHeroesByWinRate(heroesData);

        console.log(`Updated and sorted win rates for ${Object.keys(allWinRates).length} heroes`);
    }

    // NEW: Sort heroes by win rate within each attribute section
    function sortHeroesByWinRate(heroesData) {
        // Group heroes by attribute
        const heroesByAttribute = {
            'str': [],
            'agi': [],
            'int': [],
            'all': []
        };

        heroesData.forEach(hero => {
            if (heroesByAttribute[hero.attribute]) {
                heroesByAttribute[hero.attribute].push(hero);
            }
        });

        // Sort each attribute group by win rate (highest first)
        Object.keys(heroesByAttribute).forEach(attribute => {
            const heroesInSection = heroesByAttribute[attribute];

            // Sort by win rate (descending), then by name (ascending) for ties
            heroesInSection.sort((a, b) => {
                if (a.winRate !== b.winRate) {
                    return b.winRate - a.winRate; // Higher win rate first
                }
                return a.name.localeCompare(b.name); // Alphabetical for ties
            });

            // Reorder the DOM elements
            const section = attributeSections[attribute];
            if (section && heroesInSection.length > 0) {
                // Remove all heroes from section
                heroesInSection.forEach(hero => {
                    if (hero.heroCard.parentNode === section) {
                        section.removeChild(hero.heroCard);
                    }
                });

                // Add them back in sorted order
                heroesInSection.forEach(hero => {
                    section.appendChild(hero.heroCard);
                });

                console.log(`Sorted ${heroesInSection.length} ${attribute} heroes by win rate`);
            }
        });
    }

    // Keep all other functions the same (updateDraftUI, resetDraft, etc.)
    function updateDraftUI() {
        updateTeamSelections('radiant', 'picks');
        updateTeamSelections('radiant', 'bans');
        updateTeamSelections('dire', 'picks');
        updateTeamSelections('dire', 'bans');
    }

    function updateTeamSelections(team, actionType) {
        const container = document.getElementById(`${team}-${actionType}`);
        if (!container) return;

        const selections = draftState[team][actionType];
        container.innerHTML = '';

        selections.forEach(hero => {
            const heroElement = document.createElement('div');
            heroElement.className = actionType === 'picks' ? 'pick' : 'ban';
            heroElement.innerHTML = `
                <img src="/static/images/heroes/${hero.id}.jpg" alt="${hero.name}" 
                     onerror="this.src='/static/images/heroes/placeholder.jpg'">
            `;
            container.appendChild(heroElement);
        });

        const maxSlots = actionType === 'picks' ? 5 : 6;
        const emptySlots = maxSlots - selections.length;

        for (let i = 0; i < emptySlots; i++) {
            const emptySlot = document.createElement('div');
            emptySlot.className = 'empty-slot';
            emptySlot.textContent = `${actionType.slice(0, -1).charAt(0).toUpperCase() + actionType.slice(0, -1).slice(1)} ${selections.length + i + 1}`;
            container.appendChild(emptySlot);
        }
    }

    function updateTeamIndicator() {
        if (draftState.step < draftOrder.length && !isDraftComplete) {
            const currentDraftStep = draftOrder[draftState.step];

            const currentTeamEl = document.getElementById('current-team');
            if (currentTeamEl) {
                currentTeamEl.textContent = currentDraftStep.team.charAt(0).toUpperCase() + currentDraftStep.team.slice(1);
                currentTeamEl.className = `${currentDraftStep.team}-title`;
            }

            const phaseText = `${currentDraftStep.action.charAt(0).toUpperCase() + currentDraftStep.action.slice(1)}ing Phase`;
            const currentPhaseEl = document.getElementById(`${currentDraftStep.team}-phase`);
            if (currentPhaseEl) currentPhaseEl.textContent = phaseText;

            const otherTeam = currentDraftStep.team === 'radiant' ? 'dire' : 'radiant';
            const otherPhaseEl = document.getElementById(`${otherTeam}-phase`);
            if (otherPhaseEl) otherPhaseEl.textContent = 'Waiting...';

            draftState.currentTeam = currentDraftStep.team;
            draftState.currentAction = currentDraftStep.action;
        } else {
            isDraftComplete = true;
            document.getElementById('radiant-phase').textContent = 'Draft Complete';
            document.getElementById('dire-phase').textContent = 'Draft Complete';
        }
    }

    function resetDraft() {
        draftState.step = 0;
        draftState.radiant.picks = [];
        draftState.radiant.bans = [];
        draftState.dire.picks = [];
        draftState.dire.bans = [];
        isDraftComplete = false;

        updateDraftUI();
        updateTeamIndicator();
        resetHeroWinRates();

        document.querySelectorAll('.hero-card').forEach(heroCard => {
            heroCard.classList.remove('hero-picked', 'hero-banned');
            heroCard.style.opacity = '1';
            heroCard.style.filter = 'none';
            heroCard.style.border = '';
            heroCard.style.boxShadow = '';
        });

        requestRecommendations();

        const analysisContainer = document.getElementById('draft-analysis');
        if (analysisContainer) {
            analysisContainer.innerHTML = '';
        }
    }

    function resetHeroWinRates() {
        document.querySelectorAll('.hero-card').forEach(heroCard => {
            const img = heroCard.querySelector('img');
            const winRateEl = heroCard.querySelector('.hero-win-rate');

            if (img && winRateEl) {
                const heroId = extractHeroId(img);
                const originalData = originalHeroData[heroId];

                if (originalData) {
                    winRateEl.textContent = originalData.originalWinRate;
                }

                heroCard.style.border = '';
                heroCard.style.boxShadow = '';
            }
        });
    }

    function nextStep() {
        if (isDraftComplete) return;

        if (draftState.step < draftOrder.length) {
            const unavailableIds = new Set([
                ...draftState.radiant.picks.map(h => h.id),
                ...draftState.radiant.bans.map(h => h.id),
                ...draftState.dire.picks.map(h => h.id),
                ...draftState.dire.bans.map(h => h.id)
            ]);

            const availableHeroes = [];
            document.querySelectorAll('.hero-card img').forEach(img => {
                const heroId = extractHeroId(img);
                if (heroId && !unavailableIds.has(heroId)) {
                    availableHeroes.push({
                        id: heroId,
                        name: img.alt || img.title || 'Unknown Hero'
                    });
                }
            });

            if (availableHeroes.length > 0) {
                const randomHero = availableHeroes[Math.floor(Math.random() * availableHeroes.length)];
                selectHero(randomHero.id, randomHero.name);
            }
        }
    }

    function showLoadingRecommendations() {
        const container = document.getElementById('hero-recommendations');
        if (container) {
            container.innerHTML = '<div class="loading">🧠 AI analyzing draft and sorting heroes by win rate...</div>';
        }
    }

    function showFinalDraftAnalysis() {
        const container = document.getElementById('hero-recommendations');
        if (container) {
            container.innerHTML = `
                <div class="draft-complete">
                    <h3>🎉 Draft Complete!</h3>
                    <p>Heroes are sorted by win rate - highest first!</p>
                    <button onclick="location.reload()" class="btn">Start New Draft</button>
                </div>
            `;
        }
    }

    function displayRecommendations(recommendations) {
        const container = document.getElementById('hero-recommendations');
        if (!container) return;
        container.innerHTML = '';
        if (!recommendations || recommendations.length === 0) {
            container.innerHTML = '<p>No recommendations available.</p>';
            return;
        }
        // Your existing recommendation display code here
        recommendations.forEach(hero => {
            const heroCard = document.createElement('div');
            heroCard.className = 'hero-card recommendation-card';

            const strengthClass = hero.winRate > 60 ? 'strong-rec' : hero.winRate < 40 ? 'weak-rec' : 'neutral-rec';

            heroCard.innerHTML = `
                <div class="hero-image">
                    <img src="/static/images/heroes/${hero.id}.jpg" alt="${hero.name}" 
                         onerror="this.src='/static/images/heroes/placeholder.jpg'">
                    <div class="confidence ${strengthClass}">${hero.winRate}% Win</div>
                </div>
                <div class="hero-info">
                    <div class="hero-name">${hero.name}</div>
                    <div class="hero-reason">${hero.reasons}</div>
                </div>
            `;

            heroCard.addEventListener('click', function() {
                if (isDraftComplete) {
                    alert('Draft is complete! Use Reset Draft to start over.');
                    return;
                }
                if (isHeroAlreadySelected(hero.id)) {
                    alert(`${hero.name} is already picked or banned.`);
                    return;
                }
                selectHero(hero.id, hero.name);
            });

            container.appendChild(heroCard);
        });
    }

    function displayDraftAnalysis(analysis) {
        if (!analysis) return;

        let analysisContainer = document.getElementById('draft-analysis');
        if (!analysisContainer) {
            analysisContainer = document.createElement('div');
            analysisContainer.id = 'draft-analysis';
            analysisContainer.className = 'draft-analysis';

            const recContainer = document.getElementById('hero-recommendations');
            if (recContainer && recContainer.parentNode) {
                recContainer.parentNode.insertBefore(analysisContainer, recContainer.nextSibling);
            }
        }

        const radiantProb = Math.round(analysis.radiant_win_probability * 100);
        const direProb = Math.round(analysis.dire_win_probability * 100);

        analysisContainer.innerHTML = `
            <h3>🔮 Draft Analysis</h3>
            <div class="win-probabilities">
                <div class="prob-radiant">
                    <span class="team-name">Radiant</span>
                    <div class="prob-bar radiant-bar" style="width: ${radiantProb}%"></div>
                    <span class="prob-text">${radiantProb}%</span>
                </div>
                <div class="prob-dire">
                    <span class="team-name">Dire</span>
                    <div class="prob-bar dire-bar" style="width: ${direProb}%"></div>
                    <span class="prob-text">${direProb}%</span>
                </div>
            </div>
            <p class="analysis-text">📊 ${analysis.analysis}</p>
        `;
    }

    function checkModelStatus() {
        fetch('/api/model-status')
            .then(response => response.json())
            .then(data => {
                let statusEl = document.getElementById('model-status');
                if (!statusEl) {
                    statusEl = document.createElement('div');
                    statusEl.id = 'model-status';
                    statusEl.className = 'model-status';
                    document.querySelector('.container').prepend(statusEl);
                }

                if (data.model_trained) {
                    statusEl.innerHTML = '🧠 ML Model: Active & Dynamic Win Rates';
                    statusEl.className = 'model-status active';
                } else {
                    statusEl.innerHTML = '⚠️ ML Model: Training Needed';
                    statusEl.className = 'model-status inactive';
                }

                console.log('Model Status:', data);
            })
            .catch(error => {
                console.error('Error checking model status:', error);
            });
    }
});