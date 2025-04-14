document.addEventListener('DOMContentLoaded', function() {
    // Draft state
    const draftState = {
        currentTeam: 'radiant',
        currentAction: 'pick',
        step: 0,
        radiant: {
            picks: [],
            bans: []
        },
        dire: {
            picks: [],
            bans: []
        }
    };

    // Draft order (simplified Captain's Mode)
    const draftOrder = [
        { team: 'radiant', action: 'ban' },
        { team: 'dire', action: 'ban' },
        { team: 'radiant', action: 'ban' },
        { team: 'dire', action: 'ban' },
        { team: 'radiant', action: 'pick' },
        { team: 'dire', action: 'pick' },
        { team: 'dire', action: 'pick' },
        { team: 'radiant', action: 'pick' },
        { team: 'radiant', action: 'ban' },
        { team: 'dire', action: 'ban' },
        { team: 'radiant', action: 'ban' },
        { team: 'dire', action: 'ban' },
        { team: 'dire', action: 'pick' },
        { team: 'radiant', action: 'pick' },
        { team: 'radiant', action: 'pick' },
        { team: 'dire', action: 'pick' },
        { team: 'dire', action: 'ban' },
        { team: 'radiant', action: 'ban' },
        { team: 'dire', action: 'pick' },
        { team: 'radiant', action: 'pick' }
    ];

    // Initialize event listeners
    initializeEventListeners();

    // Update UI to reflect current state
    updateDraftUI();
    updateTeamIndicator();

    // Request initial recommendations
    requestRecommendations();

    function initializeEventListeners() {
        // Add click handlers to all hero grid items
        document.querySelectorAll('.hero-grid-item').forEach(heroItem => {
            heroItem.addEventListener('click', function() {
                const heroId = parseInt(this.dataset.id);
                const heroName = this.dataset.name;

                // Check if hero is already picked or banned
                if (isHeroAlreadySelected(heroId)) {
                    alert(`${heroName} is already picked or banned.`);
                    return;
                }

                // Add hero to current team's picks or bans
                selectHero(heroId, heroName);
            });
        });

        // Pick/ban toggle buttons
        document.getElementById('pick-btn').addEventListener('click', function() {
            draftState.currentAction = 'pick';
            document.getElementById('pick-btn').classList.add('active');
            document.getElementById('ban-btn').classList.remove('active');
            requestRecommendations();
        });

        document.getElementById('ban-btn').addEventListener('click', function() {
            draftState.currentAction = 'ban';
            document.getElementById('ban-btn').classList.add('active');
            document.getElementById('pick-btn').classList.remove('active');
            requestRecommendations();
        });

        // Reset button
        document.getElementById('reset-draft').addEventListener('click', resetDraft);

        // Next step button
        document.getElementById('next-step').addEventListener('click', nextStep);

        // Hero search functionality
        document.getElementById('hero-search').addEventListener('input', function() {
            const searchTerm = this.value.toLowerCase();

            document.querySelectorAll('.hero-grid-item').forEach(heroItem => {
                const heroName = heroItem.dataset.name.toLowerCase();

                if (heroName.includes(searchTerm)) {
                    heroItem.style.display = 'block';
                } else {
                    heroItem.style.display = 'none';
                }
            });
        });
    }

    function selectHero(heroId, heroName) {
        // If we're following the draft order
        if (draftState.step < draftOrder.length) {
            const currentDraftStep = draftOrder[draftState.step];
            draftState.currentTeam = currentDraftStep.team;
            draftState.currentAction = currentDraftStep.action;
        }

        // Add hero to current team's selections
        const actionType = draftState.currentAction + 's'; // 'picks' or 'bans'
        draftState[draftState.currentTeam][actionType].push({
            id: heroId,
            name: heroName
        });

        // Update UI
        updateDraftUI();

        // Move to next step in draft order
        if (draftState.step < draftOrder.length) {
            draftState.step++;

            // Update team indicator
            updateTeamIndicator();

            // Get new recommendations
            requestRecommendations();
        }
    }

    function isHeroAlreadySelected(heroId) {
        // Check radiant picks
        if (draftState.radiant.picks.some(h => h.id === heroId)) return true;

        // Check radiant bans
        if (draftState.radiant.bans.some(h => h.id === heroId)) return true;

        // Check dire picks
        if (draftState.dire.picks.some(h => h.id === heroId)) return true;

        // Check dire bans
        if (draftState.dire.bans.some(h => h.id === heroId)) return true;

        return false;
    }

    function updateDraftUI() {
        // Update radiant picks
        updateTeamSelections('radiant', 'picks');

        // Update radiant bans
        updateTeamSelections('radiant', 'bans');

        // Update dire picks
        updateTeamSelections('dire', 'picks');

        // Update dire bans
        updateTeamSelections('dire', 'bans');
    }

    function updateTeamSelections(team, actionType) {
        const container = document.getElementById(`${team}-${actionType}`);
        const selections = draftState[team][actionType];

        // Clear container
        container.innerHTML = '';

        // Add selected heroes
        selections.forEach(hero => {
            const heroElement = document.createElement('div');
            heroElement.className = actionType === 'picks' ? 'pick' : 'ban';

            // Try to load the hero image
            heroElement.innerHTML = `
                <img src="/static/images/heroes/${hero.id}.jpg" alt="${hero.name}" 
                     onerror="this.src='/static/images/heroes/placeholder.jpg'">
            `;

            container.appendChild(heroElement);
        });

        // Add empty slots
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
        if (draftState.step < draftOrder.length) {
            const currentDraftStep = draftOrder[draftState.step];

            // Update current team indicator
            document.getElementById('current-team').textContent = currentDraftStep.team.charAt(0).toUpperCase() + currentDraftStep.team.slice(1);
            document.getElementById('current-team').className = `${currentDraftStep.team}-title`;

            // Update action buttons
            if (currentDraftStep.action === 'pick') {
                document.getElementById('pick-btn').classList.add('active');
                document.getElementById('ban-btn').classList.remove('active');
            } else {
                document.getElementById('ban-btn').classList.add('active');
                document.getElementById('pick-btn').classList.remove('active');
            }

            // Update phase text
            const phaseText = `${currentDraftStep.action.charAt(0).toUpperCase() + currentDraftStep.action.slice(1)}ing Phase`;
            document.getElementById(`${currentDraftStep.team}-phase`).textContent = phaseText;

            // Reset other team's phase text
            const otherTeam = currentDraftStep.team === 'radiant' ? 'dire' : 'radiant';
            document.getElementById(`${otherTeam}-phase`).textContent = 'Waiting...';

            // Store current team and action
            draftState.currentTeam = currentDraftStep.team;
            draftState.currentAction = currentDraftStep.action;
        } else {
            // Draft complete
            document.getElementById('radiant-phase').textContent = 'Draft Complete';
            document.getElementById('dire-phase').textContent = 'Draft Complete';

            // Hide action buttons
            document.getElementById('pick-btn').disabled = true;
            document.getElementById('ban-btn').disabled = true;
        }
    }

    function requestRecommendations() {
        // If draft is complete, don't request recommendations
        if (draftState.step >= draftOrder.length) return;

        // Gather selected hero IDs
        const selectedHeroes = {
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

        // Make API request to get recommendations
        fetch('/api/counterpicks', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(selectedHeroes)
        })
            .then(response => response.json())
            .then(data => {
                displayRecommendations(data.recommendations);
            })
            .catch(error => {
                console.error('Error fetching recommendations:', error);
            });
    }

    function displayRecommendations(recommendations) {
        const container = document.getElementById('hero-recommendations');
        container.innerHTML = '';

        if (!recommendations || recommendations.length === 0) {
            container.innerHTML = '<p>No recommendations available.</p>';
            return;
        }

        recommendations.forEach(hero => {
            const heroCard = document.createElement('div');
            heroCard.className = 'hero-card';

            heroCard.innerHTML = `
                <div class="hero-image">
                    <img src="/static/images/heroes/${hero.id}.jpg" alt="${hero.name}" 
                         onerror="this.src='/static/images/heroes/placeholder.jpg'">
                    <div class="confidence">${hero.winRate}% Win</div>
                </div>
                <div class="hero-info">
                    <div class="hero-name">${hero.name}</div>
                    <div class="hero-reason">${hero.reasons}</div>
                </div>
            `;

            // Add click handler
            heroCard.addEventListener('click', function() {
                if (isHeroAlreadySelected(hero.id)) {
                    alert(`${hero.name} is already picked or banned.`);
                    return;
                }

                selectHero(hero.id, hero.name);
            });

            container.appendChild(heroCard);
        });
    }

    function resetDraft() {
        // Reset draft state
        draftState.step = 0;
        draftState.radiant.picks = [];
        draftState.radiant.bans = [];
        draftState.dire.picks = [];
        draftState.dire.bans = [];

        // Update UI
        updateDraftUI();
        updateTeamIndicator();

        // Get new recommendations
        requestRecommendations();

        // Re-enable buttons
        document.getElementById('pick-btn').disabled = false;
        document.getElementById('ban-btn').disabled = false;
    }

    function nextStep() {
        // If we're following draft order and not at the end
        if (draftState.step < draftOrder.length) {
            // Get a random hero that's not already picked or banned
            const allHeroElements = document.querySelectorAll('.hero-grid-item');
            const availableHeroes = Array.from(allHeroElements).filter(heroElem => {
                const heroId = parseInt(heroElem.dataset.id);
                return !isHeroAlreadySelected(heroId);
            });

            if (availableHeroes.length > 0) {
                // Pick a random hero from available ones
                const randomIndex = Math.floor(Math.random() * availableHeroes.length);
                const randomHero = availableHeroes[randomIndex];

                // Select this hero
                selectHero(
                    parseInt(randomHero.dataset.id),
                    randomHero.dataset.name
                );
            }
        }
    }
});