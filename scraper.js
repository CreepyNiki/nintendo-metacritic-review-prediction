// javascript
const axios = require('axios');
const cheerio = require('cheerio');

async function detectLang(text) {
    try {
        const mod = await import('eld');
        const eld = mod.default ?? mod.eld ?? mod;

        if (!eld.__eldLoaded) {
            if (typeof eld.load === 'function') {
                await eld.load();
            }
            eld.__eldLoaded = true;
        }

        if (typeof eld.detect === 'function') return eld.detect(text);
        console.warn('eld: keine passende Erkennungs-API gefunden');
    } catch (err) {
        console.warn('eld import/usage failed:', err && err.message ? err.message : err);
    }
    return null;
}

async function fetchPage(url) {
    try {
        return await axios.get(url);
    } catch (error) {
        throw error;
    }
}

async function getMetacriticData(url) {
    const response = await fetchPage(url);
    const $ = cheerio.load(response.data);

    const containers = $('.c-pageProductReviews_row .c-siteReview');
    const results = [];
    containers.each((i, el) => {
        const container = $(el);

        const username = container.find('.c-siteReviewHeader_username').text().trim();
        const rating = container.find('.c-siteReviewHeader_reviewScore').text().trim();
        const date = container.find('.c-siteReview_reviewDate').text().trim();
        const review = container.find('.c-siteReview_quote').text().trim();

        results.push({ username, rating, date, review });
    });

    return results;
}

const games = [
    'mario-kart-world',
    'pokemon-black-version',
    'the-legend-of-zelda-breath-of-the-wild',
    'nintendo-switch-2-welcome-tour',
    'new-super-mario-bros-u',
    'super-mario-galaxy-2',
    'pokemon-scarlet',
    'super-mario-bros-wonder',
    'mario-and-luigi-brothership',
    'paper-mario-sticker-star',
];

const URL = `https://www.metacritic.com/game/${games[0]}/user-reviews/`;

(async () => {
    console.log('Language Detection:', await detectLang('Hello, how are you?'));

        // const data = await getMetacriticData(URL);
        // console.log(data);
})();
