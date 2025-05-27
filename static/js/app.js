// Enhanced app.js - Updated win rate border styling
let draftState = {
    currentTeam: 'radiant',
    currentAction: 'pick',
    step: 0,
    radiant: { picks: [], bans: [] },
    dire: { picks: [], bans: [] }
};

let isCalculatingWinRates = false;
let originalHeroData = {};
let isDraftComplete = false;

// DRAFT ORDER (global)
const draftOrder = [
    { team: 'radiant', action: 'ban' }, { team: 'dire', action: 'ban' },
    { team: 'dire', action: 'ban' }, { team: 'radiant', action: 'ban' },
    { team: 'dire', action: 'ban' }, { team: 'dire', action: 'ban' },
    { team: 'radiant', action: 'ban' },
    { team: 'radiant', action: 'pick' }, { team: 'dire', action: 'pick' },
    { team: 'radiant', action: 'ban' }, { team: 'radiant', action: 'ban' },
    { team: 'dire', action: 'ban' },
    { team: 'dire', action: 'pick' }, { team: 'radiant', action: 'pick' },
    { team: 'radiant', action: 'pick' }, { team: 'dire', action: 'pick' },
    { team: 'dire', action: 'pick' }, { team: 'radiant', action: 'pick' },
    { team: 'radiant', action: 'ban' }, { team: 'dire', action: 'ban' },
    { team: 'dire', action: 'ban' }, { team: 'radiant', action: 'ban' },
    { team: 'radiant', action: 'pick' }, { team: 'dire', action: 'pick' }
];

// ATTRIBUTE SECTIONS (global)
let attributeSections = {};

function initializeEventListeners() {
    // Add click handlers to all hero cards
    document.querySelectorAll('.hero-card').forEach(heroCard => {
        heroCard.addEventListener('click', function() {
            // PREVENT CLICKS during ML calculation
            if (isCalculatingWinRates) {
                showLoadingMessage();
                return;
            }

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
    document.getElementById('reset-draft')?.addEventListener('click', function() {
        if (isCalculatingWinRates) {
            showLoadingMessage();
            return;
        }
        resetDraft();
    });

    document.getElementById('next-step')?.addEventListener('click', function() {
        if (isCalculatingWinRates) {
            showLoadingMessage();
            return;
        }
        nextStep();
    });

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
                attribute: getHeroAttribute(heroCard)
            };
        }
    });
}

function getHeroAttribute(heroCard) {
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

    // START LOADING STATE
    setLoadingState(true);
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

    // Make TWO API calls: one for recommendations, one for all hero win rates
    Promise.all([
        // Get recommendations
        fetch('/api/counterpicks', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        }).then(response => response.json()),

        // Get ALL hero win rates
        fetch('/api/hero-winrates', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        }).then(response => response.json())
    ])
        .then(([recommendationsData, winRatesData]) => {
            displayRecommendations(recommendationsData.recommendations || []);
            displayDraftAnalysis(recommendationsData.analysis);

            // Update ALL hero win rates and sort
            if (winRatesData.hero_winrates) {
                updateAllHeroWinRatesAndSort(winRatesData.hero_winrates);
            }
        })
        .catch(error => {
            console.error('Error fetching data:', error);
            displayRecommendations([]);
        })
        .finally(() => {
            // END LOADING STATE
            setLoadingState(false);
        });
}

function updateAllHeroWinRatesAndSort(allWinRates) {
    console.log('Updating win rates and sorting heroes...');

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
                    // Reset styling for unavailable heroes
                    heroCard.style.border = '';
                    heroCard.style.boxShadow = '';
                } else {
                    winRateEl.textContent = `${newWinRate}%`;

                    // Apply green border styling
                    if (newWinRate >= 56) {
                        // Strong pick - bright green border with stronger glow
                        heroCard.style.border = '2px solid #4CAF50';
                        heroCard.style.boxShadow = '0 0 15px rgba(76, 175, 80, 0.6)';
                    } else if (newWinRate >= 51) {
                        // Good pick - lighter green border with subtle glow
                        heroCard.style.border = '2px solid #81C784';
                        heroCard.style.boxShadow = '0 0 8px rgba(129, 199, 132, 0.4)';
                    } else {
                        // Average or below - no special border
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

    // Sort and reorder heroes within each attribute section
    sortHeroesByWinRate(heroesData);

    console.log(`✅ Updated and sorted win rates for ${Object.keys(allWinRates).length} heroes`);
}

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
// Update the HTML template to match - you'll need 6 ban slots per team
function updateTeamSelections(team, actionType) {
    const container = document.getElementById(`${team}-${actionType}`);
    if (!container) return;

    const selections = draftState[team][actionType];

    // Get all slots (both filled and empty) from your HTML template
    const allSlots = container.querySelectorAll('.pick, .ban, .empty-slot');

    // Fill in the selections without destroying your draft order numbers
    selections.forEach((hero, index) => {
        if (allSlots[index]) {
            // Replace empty slot with hero
            const heroElement = document.createElement('div');
            heroElement.className = actionType === 'picks' ? 'pick' : 'ban';
            heroElement.innerHTML = `
                <img src="/static/images/heroes/${hero.id}.jpg" alt="${hero.name}" 
                     onerror="this.src='/static/images/heroes/placeholder.jpg'">
            `;

            // Replace the empty slot with the hero element
            allSlots[index].replaceWith(heroElement);
        }
    });
}
// Update the phase indicator to be more descriptive
function updateTeamIndicator() {
    if (draftState.step < draftOrder.length && !isDraftComplete) {
        const currentDraftStep = draftOrder[draftState.step];

        const currentTeamEl = document.getElementById('current-team');
        if (currentTeamEl) {
            currentTeamEl.textContent = currentDraftStep.team.charAt(0).toUpperCase() + currentDraftStep.team.slice(1);
            currentTeamEl.className = `${currentDraftStep.team}-title`;
        }

        // More descriptive phase text based on draft position
        let phaseText = '';
        const step = draftState.step + 1;

        if (step <= 7) {
            phaseText = `Initial ${currentDraftStep.action.charAt(0).toUpperCase() + currentDraftStep.action.slice(1)}`;
        } else if (step <= 9) {
            phaseText = `First ${currentDraftStep.action.charAt(0).toUpperCase() + currentDraftStep.action.slice(1)}`;
        } else if (step <= 12) {
            phaseText = `Mid ${currentDraftStep.action.charAt(0).toUpperCase() + currentDraftStep.action.slice(1)}`;
        } else if (step <= 18) {
            phaseText = `Core ${currentDraftStep.action.charAt(0).toUpperCase() + currentDraftStep.action.slice(1)}`;
        } else if (step <= 22) {
            phaseText = `Final ${currentDraftStep.action.charAt(0).toUpperCase() + currentDraftStep.action.slice(1)}`;
        } else {
            phaseText = `Last ${currentDraftStep.action.charAt(0).toUpperCase() + currentDraftStep.action.slice(1)}`;
        }

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
// Call validation after each selection
function selectHero(heroId, heroName) {
    if (isDraftComplete || isCalculatingWinRates) return;

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

    // Update UI immediately
    updateDraftUI();
    updateHeroAvailability(heroId, false);

    // Move to next step
    draftState.step++;

    // CHECK IF DRAFT IS COMPLETE
    if (draftState.step >= draftOrder.length) {
        isDraftComplete = true;
        console.log('🎉 Draft completed!');

        // Update team indicators
        document.getElementById('radiant-phase').textContent = 'Draft Complete';
        document.getElementById('dire-phase').textContent = 'Draft Complete';

        // Show final analysis after a short delay
        setTimeout(() => {
            showFinalDraftAnalysis();
        }, 500);

        return; // Don't continue with recommendations
    }

    // Continue with normal flow if draft not complete
    updateTeamIndicator();

    // Request recommendations for next step
    setTimeout(() => {
        requestRecommendations();
    }, 100);
}

function resetDraft() {
    draftState.step = 0;
    draftState.radiant.picks = [];
    draftState.radiant.bans = [];
    draftState.dire.picks = [];
    draftState.dire.bans = [];
    isDraftComplete = false;

    // Reset to original template state - restore draft order numbers
    resetToOriginalTemplate();

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
        }

        heroCard.style.border = '';
        heroCard.style.boxShadow = '';
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
        container.innerHTML = `
            <div class="loading">
                <div class="loading-spinner"></div>
                <p>🧠 AI calculating win rates for all heroes...</p>
                <p><small>Please wait, this may take a few seconds</small></p>
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

function showLoadingMessage() {
    // Create or update loading toast
    let loadingToast = document.getElementById('loading-toast');
    if (!loadingToast) {
        loadingToast = document.createElement('div');
        loadingToast.id = 'loading-toast';
        loadingToast.className = 'loading-toast';
        document.body.appendChild(loadingToast);
    }

    loadingToast.textContent = '🧠 Please wait, AI is calculating win rates...';
    loadingToast.classList.add('show');

    // Auto-hide after 2 seconds
    setTimeout(() => {
        loadingToast.classList.remove('show');
    }, 2000);
}
// Enable/disable all interactive elements
function setLoadingState(isLoading) {
    isCalculatingWinRates = isLoading;

    // Update cursor and visual feedback
    document.body.classList.toggle('calculating', isLoading);

    // Update hero cards
    document.querySelectorAll('.hero-card').forEach(heroCard => {
        if (isLoading) {
            heroCard.classList.add('disabled');
            heroCard.style.cursor = 'wait';
        } else {
            heroCard.classList.remove('disabled');
            heroCard.style.cursor = 'pointer';
        }
    });

    // Update buttons
    const buttons = document.querySelectorAll('button');
    buttons.forEach(button => {
        button.disabled = isLoading;
        if (isLoading) {
            button.style.cursor = 'wait';
            button.style.opacity = '0.6';
        } else {
            button.style.cursor = 'pointer';
            button.style.opacity = '1';
        }
    });
}

function showFinalDraftAnalysis() {
    console.log('📊 Showing final draft analysis...');

    // Find or create the recommendations container
    let container = document.getElementById('hero-recommendations');

    if (!container) {
        // Create the container in the right sidebar
        const sidebar = document.querySelector('.draft-ui-wrapper');
        if (sidebar) {
            container = document.createElement('div');
            container.id = 'hero-recommendations';
            container.style.backgroundColor = 'var(--panel-color)';
            container.style.borderRadius = '8px';
            container.style.padding = '15px';
            container.style.marginTop = '20px';
            sidebar.appendChild(container);
        } else {
            // Fallback: create a modal overlay
            container = document.createElement('div');
            container.style.position = 'fixed';
            container.style.top = '0';
            container.style.left = '0';
            container.style.width = '100%';
            container.style.height = '100%';
            container.style.backgroundColor = 'rgba(0,0,0,0.8)';
            container.style.display = 'flex';
            container.style.justifyContent = 'center';
            container.style.alignItems = 'center';
            container.style.zIndex = '10000';
            document.body.appendChild(container);
        }
    }

    if (container) {
        // Get final draft prediction
        getFinalDraftPrediction().then(predictionData => {
            const { radiant_win_probability, dire_win_probability, analysis, detailed_analysis } = predictionData;

            // Determine winning team
            const radiantPercent = Math.round(radiant_win_probability * 100);
            const direPercent = Math.round(dire_win_probability * 100);
            const winningTeam = radiant_win_probability > dire_win_probability ? 'radiant' : 'dire';
            const winningPercent = winningTeam === 'radiant' ? radiantPercent : direPercent;
            const confidence = Math.abs(radiantPercent - direPercent);

            // Get confidence level
            let confidenceText = '';
            let confidenceIcon = '';
            if (confidence >= 20) {
                confidenceText = 'High Confidence';
                confidenceIcon = '🎯';
            } else if (confidence >= 10) {
                confidenceText = 'Medium Confidence';
                confidenceIcon = '⚖️';
            } else {
                confidenceText = 'Low Confidence - Very Close Draft';
                confidenceIcon = '🤔';
            }

            container.innerHTML = `
                <div class="final-draft-analysis">
                    <h2>🏆 Final Draft Analysis</h2>
                    
                    <div class="win-prediction-card">
                        <div class="prediction-header">
                            <h3>${confidenceIcon} Prediction: <span class="${winningTeam}-color">${winningTeam.toUpperCase()}</span> Favored</h3>
                            <div class="confidence-badge ${confidence >= 15 ? 'high' : confidence >= 8 ? 'medium' : 'low'}">
                                ${confidenceText}
                            </div>
                        </div>
                        
                        <div class="probability-display">
                            <div class="team-probability radiant ${winningTeam === 'radiant' ? 'winner' : ''}">
                                <div class="team-name">RADIANT</div>
                                <div class="probability-number">${radiantPercent}%</div>
                                <div class="probability-bar">
                                    <div class="bar radiant-bar" style="width: ${radiantPercent}%"></div>
                                </div>
                            </div>
                            
                            <div class="vs-divider">VS</div>
                            
                            <div class="team-probability dire ${winningTeam === 'dire' ? 'winner' : ''}">
                                <div class="team-name">DIRE</div>
                                <div class="probability-number">${direPercent}%</div>
                                <div class="probability-bar">
                                    <div class="bar dire-bar" style="width: ${direPercent}%"></div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="analysis-summary">
                            <p><strong>📊 Analysis:</strong> ${analysis}</p>
                            ${detailed_analysis ? `<p><strong>💡 Details:</strong> ${detailed_analysis}</p>` : ''}
                        </div>
                    </div>
                    
                    <div class="draft-summary">
                        <h4>📋 Draft Summary</h4>
                        <div class="team-summary">
                            <div class="team-comp radiant-comp">
                                <h5>Radiant Composition</h5>
                                <div class="hero-lineup">
                                    ${getDraftSummaryHTML('radiant')}
                                </div>
                            </div>
                            <div class="team-comp dire-comp">
                                <h5>Dire Composition</h5>
                                <div class="hero-lineup">
                                    ${getDraftSummaryHTML('dire')}
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="action-buttons">
                        <button onclick="location.reload()" class="btn primary">🔄 Start New Draft</button>
                    </div>
                </div>
            `;
        }).catch(error => {
            console.error('Error getting final prediction:', error);
            // Fallback display
            container.innerHTML = `
                <div class="final-draft-analysis">
                    <h2>🎉 Draft Complete!</h2>
                    <p>Draft completed successfully!</p>
                    <button onclick="location.reload()" class="btn primary">🔄 Start New Draft</button>
                </div>
            `;
        });
    }
}

// Get final prediction from the ML model
async function getFinalDraftPrediction() {
    console.log('Getting final prediction...');

    const finalDraftData = {
        radiant: {
            picks: draftState.radiant.picks.map(h => h.id),
            bans: draftState.radiant.bans.map(h => h.id)
        },
        dire: {
            picks: draftState.dire.picks.map(h => h.id),
            bans: draftState.dire.bans.map(h => h.id)
        },
        currentTeam: 'radiant',
        currentAction: 'pick'
    };

    console.log('Final draft data:', finalDraftData);

    const response = await fetch('/api/counterpicks', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(finalDraftData)
    });

    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    console.log('API response:', data);

    return data.analysis;
}

// Generate HTML for team composition summary
function getDraftSummaryHTML(team) {
    const picks = draftState[team].picks;

    if (!picks || picks.length === 0) {
        return '<p>No heroes selected</p>';
    }

    return picks.map(hero => `
        <div class="hero-summary">
            <img src="/static/images/heroes/${hero.id}.jpg" alt="${hero.name}" 
                 onerror="this.src='/static/images/heroes/placeholder.jpg'">
            <span class="hero-name">${hero.name}</span>
        </div>
    `).join('');
}
document.addEventListener('DOMContentLoaded', function() {

    // Store attribute section containers for sorting
    attributeSections = {
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

});