// javascript
const puppeteer = require('puppeteer');

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

async function extractReviewsFromPage(page) {
    return await page.evaluate(() => {
        const nodes = Array.from(document.querySelectorAll('.c-pageProductReviews_row .c-siteReview'));
        return nodes.map(n => {
            const username = n.querySelector('.c-siteReviewHeader_username')?.textContent?.trim() || '';
            const ratingText = n.querySelector('.c-siteReviewHeader_reviewScore')?.textContent?.trim() || '';
            const rating = parseInt(ratingText.replace(/[^0-9]/g, ''), 10) || 0;
            const date = n.querySelector('.c-siteReview_reviewDate')?.textContent?.trim() || '';
            const review = n.querySelector('.c-siteReview_quote')?.textContent?.trim() || '';
            return { username, rating, date, review };
        });
    });
}

async function scrollToBottom(page) {
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
}

(async () => {
    const browser = await puppeteer.launch();
    const page = await browser.newPage();
    await page.goto(URL);

    const collected = [];
    let negativeReviews = 0;
    let neutralReviews = 0;
    let positiveReviews = 0;

    while (!(negativeReviews >= 34 && neutralReviews >= 33 && positiveReviews >= 33)) {
        
        const reviews = await extractReviewsFromPage(page);
        
        for (const review of reviews) {

            const lang = await detectLang(review.review || '');
            if (lang.language === 'en') {

                if (review.rating < 4) {
                    if (negativeReviews <= 33) {
                        negativeReviews++;
                        collected.push(review);
                    }
                } else if (review.rating > 7) {
                    if (positiveReviews < 33) {
                        positiveReviews++;
                        collected.push(review);
                    }
                } else {
                    if (neutralReviews < 33) {
                        neutralReviews++;
                        collected.push(review);
                    }
                }

                if (negativeReviews >= 34 && neutralReviews >= 33 && positiveReviews >= 33) break;
            }
        }

        await scrollToBottom(page);
    }

    console.log('Gefundene Reviews:', collected.length);
    console.log('Negative:', negativeReviews, 'Neutral:', neutralReviews, 'Positive:', positiveReviews);
    console.log(collected);

    await browser.close();
})();
