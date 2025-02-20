import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import './style.css';
import { Protocol } from 'pmtiles';
import { searchAddresses, AddressDatabase } from './addresses.js';

const DB_INIT_MESSAGE = 'Initializing addresses. This may take a few seconds. Future searches will be much quicker.';

function getZipFromUrl() {
    const params = new URLSearchParams(window.location.search);
    return params.get('zip');
}

function setZipInUrl(zip) {
    const url = new URL(window.location);
    if (zip) {
        url.searchParams.set('zip', zip);
    } else {
        url.searchParams.delete('zip');
    }
    window.history.pushState({}, '', url);
}

const addressDB = new AddressDatabase();
let isDBInitialized = false;
let currentAddresses = [];

if (!maplibregl) {
    alert('Error loading MapLibre GL JS. Please check your internet connection.');
}

// Manually Prefix PMTILES_URL with import.meta.env.BASE_URL as defined in vite.config.js
const PMTILES_URL = import.meta.env.BASE_URL + "/roofs.pmtiles";

let protocol = new Protocol();
maplibregl.addProtocol("pmtiles", protocol.tile);

const map = new maplibregl.Map({
    container: 'map',
    style: {
        version: 8,
        glyphs: "https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf",
        sources: {
            swissimage: {
                type: 'raster',
                tiles: [
                    'https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.swissimage/default/current/3857/{z}/{x}/{y}.jpeg'
                ],
                tileSize: 256,
                attribution: 'swissIMAGE © swisstopo',
                maxzoom: 19
            },
            roofs: {
                id: "id",
                type: "vector",
                url: "pmtiles://" + PMTILES_URL,
                attribution: "swissBuildings3D © swisstopo | addresses © BFS"
            }
        },
        layers: [
            {
                id: "swissimage",
                type: "raster",
                source: "swissimage",
                minzoom: 0,
                maxzoom: 21,
                paint: {
                    "raster-opacity": [
                        "interpolate",
                        ["linear"],
                        ["zoom"],
                        10, 0.3,
                        14, 1.0
                    ]
                }
            },
            {
                id: "roofs-fill",
                type: "fill",
                source: "roofs",
                "source-layer": "roof",
                paint: {
                    "fill-color": [
                        "case",
                        ["to-boolean", ["feature-state", "highlighted"]],
                        "#ffff00",
                        "#8b0000"
                    ],
                    "fill-opacity": 0.5
                }
            },
            {
                id: "roofs-outline",
                type: "line",
                source: "roofs",
                "source-layer": "roof",
                paint: {
                    "line-color": "#000000",
                    "line-width": 1
                }
            },
            {
                id: "roofs-label",
                type: "symbol",
                source: "roofs",
                "source-layer": "roof",
                minzoom: 17,
                layout: {
                    "text-field": [
                        "concat",
                        "↑",
                        ["to-string", ["number-format", ["get", "roof_min"], { "min-fraction-digits": 1, "max-fraction-digits": 1 }]],
                        "m\n",
                        "↓",
                        ["to-string", ["number-format", ["get", "roof_max"], { "min-fraction-digits": 1, "max-fraction-digits": 1 }]],
                        "m"
                    ],
                    "text-size": 14,
                    "text-allow-overlap": false,
                    "text-ignore-placement": false,
                    "text-optional": true,
                    "text-anchor": "center",
                    "text-max-width": 8,
                    "text-line-height": 1.2
                },
                paint: {
                    "text-color": "#000000",
                    "text-halo-color": "#ffffff",
                    "text-halo-width": 1.5
                }
            }
        ]
    },
    center: [8.2275, 46.8182], 
    zoom: 8
});

let highlightedId = null;

// Show building properties on click
map.on('click', 'roofs-fill', (e) => {
    if (!e.features.length) return;

    const feature = e.features[0];
    const properties = feature.properties;
    const clickedId = feature.id;
    
    // Remove highlight from previously highlighted feature
    if (highlightedId !== null) {
        map.setFeatureState(
            { source: 'roofs', id: highlightedId, sourceLayer: "roof" },
            { highlighted: false }
        );
    }

    // Highlight the clicked feature
    if (highlightedId !== clickedId) {
        map.setFeatureState(
            { source: 'roofs', id: clickedId, sourceLayer: "roof"},
            { highlighted: true }
        );
        highlightedId = clickedId;
    } else {
        highlightedId = null;
    }
    
    let html = '<table id="property-table">';
    if (properties.roof_max !== undefined) {
        html += `<tr><td>Max Height:</td><td>${properties.roof_max.toFixed(1)}m</td></tr>`;
    }
    if (properties.roof_min !== undefined) {
        html += `<tr><td>Min Height:</td><td>${properties.roof_min.toFixed(1)}m</td></tr>`;
    }
    if (properties.year) {
        html += `<tr><td>Year of Digitization</td><td>${properties.year}</td></tr>`;
    }
    if (properties.objecttype) {
        html += `<tr><td>Building Type</td><td>${properties.objecttype}</td></tr>`;
    }
    html += '</table>';

    document.getElementById('feature-properties').innerHTML = html;
});

map.on('error', (e) => {
    console.error('Map error:', e);
});

map.on('mouseenter', 'roofs-fill', () => {
    map.getCanvas().style.cursor = 'pointer';
});

map.on('mouseleave', 'roofs-fill', () => {
    map.getCanvas().style.cursor = '';
});

// Clear highlight when clicking outside of buildings
map.on('click', (e) => {
    const features = map.queryRenderedFeatures(e.point, { layers: ['roofs-fill'] });
    if (features.length === 0 && highlightedId !== null) {
        if (highlightedId !== null) {
            map.setFeatureState(
               { source: 'roofs', sourceLayer: 'roof', id: highlightedId },
                { highlighted: false }
            );
            highlightedId = null;
            document.getElementById('feature-properties').innerHTML = '';
        }
    }
});

// Add navigation controls
map.addControl(new maplibregl.NavigationControl());
map.addControl(new maplibregl.ScaleControl());

let addressMarkers = [];

function showSpinner(show, type = 'search') {
    document.getElementById(`${type}-text`).style.display = show ? 'none' : 'inline';
    document.getElementById(`${type}-spinner`).style.display = show ? 'inline' : 'none';
    document.getElementById(`${type}-btn`).disabled = show;
    if (type === 'search') {
        document.getElementById('zip-search').disabled = show;
    }
}

async function handleRandomSearch() {
    console.log("handle random search");
    try {
        document.getElementById('search-buttons').classList.add('visible');
        showSpinner(true, 'random');
        
        // Initialize the database if not already initialized
        if (!isDBInitialized) {
            document.getElementById('search-message').textContent = DB_INIT_MESSAGE;
            await addressDB.initialize();
            isDBInitialized = true;
            document.getElementById('search-message').textContent = '';
        }
        
        const randomZip = await addressDB.getRandomZipCode();
        document.getElementById('zip-search').value = randomZip;
        setZipInUrl(randomZip);
        await searchAndDisplayAddresses(randomZip);
    } catch (error) {
        console.error('Error with random search:', error);
    } finally {
        showSpinner(false, 'random');
    }
}

async function searchAndDisplayAddresses(zipCode) {
    try {
        showSpinner(true);
        
        // Initialize the database if not already initialized, this may take
        // some time as duckdb needs to be loaded into the client
        if (!isDBInitialized) {
            document.getElementById('search-buttons').classList.add('visible');  // Make sure container is visible
            document.getElementById('search-message').textContent = DB_INIT_MESSAGE;
            await addressDB.initialize();
            isDBInitialized = true;
            document.getElementById('search-message').textContent = '';
        }
        
        // Clear previous markers
        addressMarkers.forEach(marker => marker.remove());
        addressMarkers = [];
        
        const addresses = await searchAddresses(zipCode, addressDB);

        // Add index to each address
        currentAddresses = addresses.map((addr, index) => ({...addr, index}));
        
        // Show or hide address search based on results
        toggleAddressSearch(currentAddresses.length > 0);
        
        // Show message if no addresses found
        const messageEl = document.getElementById('search-message');
        if (currentAddresses.length === 0) {
            messageEl.textContent = `No addresses found for ZIP code ${zipCode}`;
            return;
        } else {
            messageEl.textContent = ''; // Clear message when addresses are found
        }
        
        const bounds = new maplibregl.LngLatBounds();
        
        currentAddresses.forEach(addr => {
            const { STRNAME, DEINR, DPLZNAME, latitude, longitude } = addr;


            if (!isNaN(latitude) && !isNaN(longitude) && latitude > 0 && longitude > 0) {
                // Create a circle marker
                const el = document.createElement('div');
                el.className = 'address-marker';
                
                const marker = new maplibregl.Marker({ element: el })
                    .setLngLat([longitude, latitude])
                      .setPopup(new maplibregl.Popup().setText(`${STRNAME} ${DEINR}, ${DPLZNAME}`))
                    .addTo(map);
                
                marker.addressIndex = addr.index;  // Store index on marker
                addressMarkers.push(marker);
                bounds.extend([longitude, latitude]);
            }
            else {
                console.log(`${STRNAME} ${DEINR} ${DPLZNAME} has no valid coordinates`);
            }
        });
        
        updateAddressCount(addressMarkers.length);

        // Zoom to the extent of all markers
        if (addressMarkers.length > 0) {
            map.fitBounds(bounds, {
                padding: 50,
                maxZoom: 15
            });
            messageEl.textContent = ''; // Clear message when addresses are found
        }
        else {
            messageEl.textContent = `Unfortunately, none of the found addresses have valid coordinates in the official dataset.`;
        }
    } catch (error) {
        console.error('Error searching addresses:', error);
    } finally {
        showSpinner(false);
    }
}

// Modify the search event handlers
function handleSearch() {
    const zipCode = document.getElementById('zip-search').value.trim();
    
    if (zipCode && /^\d{4}$/.test(zipCode)) {  // Check for 4-digit ZIP code
        setZipInUrl(zipCode);
        searchAndDisplayAddresses(zipCode);
    } else {
        // Clear markers for empty or invalid ZIP code
        addressMarkers.forEach(marker => marker.remove());
        addressMarkers = [];
        setZipInUrl(null); // Remove zip from URL
    }
}

// Check for ZIP code in URL on page load
const urlZip = getZipFromUrl();
if (urlZip && /^\d{4}$/.test(urlZip)) {
    document.getElementById('zip-search').value = urlZip;
    searchAndDisplayAddresses(urlZip);
}

// Add event listeners for the search and random buttons
document.getElementById('search-btn').addEventListener('click', handleSearch);
document.getElementById('random-btn').addEventListener('click', handleRandomSearch);

// Add event listener for Enter key in search input
document.getElementById('zip-search').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        handleSearch();
    }
});

// Show buttons on focus
document.getElementById('zip-search').addEventListener('focus', () => {
    document.getElementById('search-buttons').classList.add('visible');
});


// Update ZIP search input handler
document.getElementById('zip-search').addEventListener('input', (e) => {
    const zipCode = e.target.value.trim();
    if (!zipCode || !/^\d{4}$/.test(zipCode)) {
        // Clear markers if ZIP code is empty or invalid
        addressMarkers.forEach(marker => marker.remove());
        addressMarkers = [];
        document.getElementById('search-message').textContent = '';
        document.getElementById('street-search').value = ''; // Clear street search input
        document.getElementById('street-search-message').textContent = '';
        toggleAddressSearch(false);  // Hide street search
    }
});

// Add focus handler for ZIP search
document.getElementById('zip-search').addEventListener('focus', () => {
    // Clear street search when ZIP input is selected
    document.getElementById('street-search').value = '';
    addressMarkers.forEach(marker => marker.addTo(map));
    document.getElementById('street-search-message').textContent = '';
});

// Update search button click handler
document.getElementById('search-btn').addEventListener('click', () => {
    // Clear street search when search button is clicked
    document.getElementById('street-search').value = '';
    handleSearch();
});

function filterAddressesByStreet(addresses, searchText) {
    searchText = searchText.toLowerCase();
    return addresses.filter(addr => {
        const streetAndNumber = `${addr.STRNAME} ${addr.DEINR}`.toLowerCase();
        return streetAndNumber.includes(searchText);
    });
}

function toggleAddressSearch(show) {
    const streetSearchWrapper = document.getElementById('street-search-wrapper');
    streetSearchWrapper.style.display = show ? 'block' : 'none';
    if (!show) {
        document.querySelector('#street-search').value = '';
        document.getElementById('street-search-message').textContent = '';
    }
}

function handleStreetSearch(shouldZoom = false) {
    const searchText = document.getElementById('street-search').value.trim();
    if (!searchText) {
        // Show all markers for the ZIP code
        addressMarkers.forEach(marker => marker.addTo(map));
        updateAddressCount(currentAddresses.length);
        return;
    }

    const visibleAddresses = filterAddressesByStreet(currentAddresses, searchText);    
    const visibleIndices = new Set(visibleAddresses.map(addr => addr.index));
    
    // Create bounds for visible addresses
    const bounds = new maplibregl.LngLatBounds();
    let hasValidMarkers = false;

    addressMarkers.forEach(marker => {
        if (visibleIndices.has(marker.addressIndex)) {
            marker.addTo(map);
            const lngLat = marker.getLngLat();
            bounds.extend([lngLat.lng, lngLat.lat]);
            hasValidMarkers = true;
        } else {
            marker.remove();
        }
    });

    updateAddressCount(currentAddresses.length, visibleAddresses.length);

    // Zoom to visible addresses if requested and there are valid markers
    if (shouldZoom && hasValidMarkers) {
        map.fitBounds(bounds, {
            padding: 50,
            maxZoom: 18
        });
    }
}

// Add event listeners for street search
//document.getElementById('street-search-btn').addEventListener('click', handleStreetSearch);
document.getElementById('street-search').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        handleStreetSearch(true); // Pass true to trigger zooming
    }
});

// Add input event listener for real-time street filtering
document.getElementById('street-search').addEventListener('input', (e) => {
    handleStreetSearch(false); // Pass false to skip zooming
});

// Initialize clear buttons functionality
function initializeClearButtons() {
    document.querySelectorAll('.clear-input').forEach(button => {
        button.addEventListener('click', (e) => {
            const input = e.target.previousElementSibling;
            input.value = '';
            input.focus();
            
            // Trigger appropriate events based on which input was cleared
            if (input.id === 'zip-search') {
                // Clear markers and messages
                addressMarkers.forEach(marker => marker.remove());
                addressMarkers = [];
                document.getElementById('search-message').textContent = '';
                document.getElementById('street-search').value = '';
                toggleAddressSearch(false);
                setZipInUrl(null); // Remove zip from URL
            } else if (input.id === 'street-search') {
                handleStreetSearch();
            }
        });
    });
}

// Call this function when the page loads
initializeClearButtons();

function updateAddressCount(total, filtered = null) {
    const cityName = currentAddresses.length > 0 ? currentAddresses[0].DPLZNAME : '';
    const zip = document.getElementById('zip-search').value;
    const message = filtered !== null 
        ? `Found ${filtered} of ${total} addresses with coordinates in ${zip} ${cityName}.`
          : `Found ${total} addresses with coodinates in ${zip} ${cityName}.`;
    document.getElementById('street-search-message').textContent = message;
}

// Add cleanup when the page is unloaded
window.addEventListener('beforeunload', async () => {
    if (isDBInitialized) {
        await addressDB.close();
    }
});

document.getElementById('toggle-info').addEventListener('click', function() {
    const infoContent = document.getElementById('info-content');
    const currentDisplay = window.getComputedStyle(infoContent).display;
    if (currentDisplay === 'none') {
        infoContent.style.display = 'inline';
        this.textContent = '▼ click to hide info text';
    } else {
        infoContent.style.display = 'none';
        this.textContent = '► click to show info text';
    }
});
