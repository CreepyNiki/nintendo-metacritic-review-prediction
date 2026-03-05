const puppeteer = require('puppeteer');
const fs = require('fs');

const gameID = 9;

const limit = 40;

const games = [
    'mario-kart-world',
    'animal-crossing-new-horizons',
    'the-legend-of-zelda-breath-of-the-wild',
    'pokemon-legends-z-a',
    'nintendo-switch-sports',
    'the-legend-of-zelda-tears-of-the-kingdom',
    'pokemon-scarlet',
    'paper-mario-sticker-star',
    'super-mario-party',
    'super-smash-bros-ultimate',
];

const URL = `https://www.metacritic.com/game/${games[gameID]}/user-reviews/`;

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

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

async function extractReviewsFromPage(page) {
        return await page.evaluate(() => {

            function getUsername(el) {
                if (!el) return '';
                for (const node of el.childNodes) {
                    if (node.nodeType === Node.TEXT_NODE) {
                        const txt = (node.textContent || '').replace(/\s+/g, ' ').trim();
                        if (txt) return txt;
                    }
                }
            }

            const cards = Array.from(document.querySelectorAll('.review-card'));
            return cards.map(card => {
                const header = card.querySelector('.review-card__header') || card.querySelector('a.review-card__header') || card;
                const username = getUsername(header) || '';
                const ratingText = card.querySelector('.c-siteReviewHeader_reviewScore, .c-siteReviewScore, .c-siteReviewScore_background')?.textContent || '';
                const rating = parseInt((ratingText.match(/\d+/) || ['0'])[0], 10) || 0;
                const date = card.querySelector('.review-card__date, .c-siteReview_reviewDate')?.textContent?.trim() || '';
                const review = card.querySelector('.review-card__quote, .c-siteReview_quote, .review-body, .review_body')?.textContent?.trim() || '';
                return { username, rating, date, review };
            });
        });
}

async function getMetadata(userPage) {
        const browser = await puppeteer.launch();
        const page = await browser.newPage();
        await page.goto(userPage, { waitUntil: 'networkidle2' });
        const averageUserScore = await page.evaluate(() => {
            const scoreText = document.querySelector('.c-scoreOverview_avgScoreText')?.textContent?.trim() || '';
            return parseFloat(scoreText) || null;
        });
        const games = await page.evaluate(() => {
            const gameText = document.querySelector('.c-globalHeader_menu_subText');
            return gameText ? gameText.textContent.trim() : '';
        });
    const scoreCountTexts = await page.evaluate(() =>
        Array.from(document.querySelectorAll('.c-scoreCount_count')).map(n => (n?.textContent || '').trim())
    );
    const scoreCount = scoreCountTexts.map(t => {
        const normalized = t.replace(`^\d+`, '');
        return normalized ? parseInt(normalized, 10) : null;
    });

    const scoreCounts = {
        positive: scoreCount[0] ?? 0,
        neutral:  scoreCount[1] ?? 0,
        negative: scoreCount[2] ?? 0,
    };

        await browser.close();
        return { averageUserScore, games, scoreCounts };
}

async function scrollToBottom(page) {
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
}

async function collectReviews(metadata = true) {
    const browser = await puppeteer.launch();
    const page = await browser.newPage();
    await page.goto(URL, { waitUntil: 'domcontentloaded' });

    const collected = [];
    let negativeReviews = 0;
    let neutralReviews = 0;
    let positiveReviews = 0;

    while (!(negativeReviews >= limit && neutralReviews >= limit && positiveReviews >= limit)) {

        const reviews = await extractReviewsFromPage(page);

        for (const review of reviews) {

             const lang = await detectLang(review.review || '');
             if (lang.language === 'en') {

                 console.log(`Überprüfe Review von ${review.username} (Rating: ${review.rating}, Sprache: ${lang.language})`);
                 if (collected.some(r => (r.username === review.username))) continue;

                 if(review.review === '[SPOILER ALERT: This review contains spoilers.]') continue;

                if (review.rating < 4) {
                    if (negativeReviews < limit) {
                        negativeReviews++;
                        if(metadata === true) {
                            console.log(`Review von ${review.username}`);
                            let userPage = `https://www.metacritic.com/user/${review.username}`;
                            const metadata = await getMetadata(userPage);
                            collected.push({ ...review, metadata });
                        }else {
                            collected.push(review);
                        }
                    }
                } else if (review.rating > 7) {
                    if (positiveReviews < limit) {
                        positiveReviews++;
                        if(metadata === true) {
                            console.log(`Review von ${review.username}`);
                            let userPage = `https://www.metacritic.com/user/${review.username}`;
                            const metadata = await getMetadata(userPage);
                            collected.push({ ...review, metadata });
                        }else {
                            collected.push(review);
                        }
                    }
                } else {
                    if (neutralReviews < limit) {
                        neutralReviews++;
                        if(metadata === true) {
                            console.log(`Review von ${review.username}`);
                            let userPage = `https://www.metacritic.com/user/${review.username}`;
                            const metadata = await getMetadata(userPage);
                            collected.push({ ...review, metadata });
                        }else {
                            collected.push(review);
                        }
                    }
                }

                console.log(`Aktueller Stand - Negative: ${negativeReviews}, Neutral: ${neutralReviews}, Positive: ${positiveReviews}`);
                 if (negativeReviews >= limit && neutralReviews >= limit && positiveReviews >= limit) break;
             }
         }

        await scrollToBottom(page);

        await sleep(3000);
     }

     console.log('Gefundene Reviews:', collected.length);
     console.log('Negative:', negativeReviews, 'Neutral:', neutralReviews, 'Positive:', positiveReviews);
     await browser.close();

     const jsonFilePath = `../data/${games[gameID]}${metadata ? '_with_metadata' : ''}.json`;
     fs.writeFileSync(jsonFilePath, JSON.stringify(collected, null, 2));
     console.log(`Daten in ${jsonFilePath} gespeichert.`);

 }

 collectReviews(true);
