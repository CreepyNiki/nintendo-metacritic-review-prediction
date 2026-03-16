const puppeteer = require('puppeteer');
const fs = require('fs');

// Aktuell ausgewählte gameID, die stellvertretend für das ausgewählte Spiel steht.
const gameID = 0;

// Anzahl der gesammelten Reviews pro Kategorie (negativ, neutral, positiv).
const limit = 40;

// Ausgewählte Spiele.
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
// URL der User-Reviews-Seite des ausgewählten Spiels auf Metacritic.
const URL = `https://www.metacritic.com/game/${games[gameID]}/user-reviews/`;

// Hilfsfunktion, die eine Pause von ms Millisekunden einlegt, um die Seite nicht zu überlasten und sicherzustellen, dass alle Inhalte geladen sind, bevor fortgefahren wird.
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

// Funktion, die für alle Reviews die geschriebene Sprache herausfiltert. -> https://www.npmjs.com/package/eld
async function detectLang(text) {
    const mod = await import('eld');
    const eld = mod.default;
    // Datenbank wird geladen.
    await eld.load();
    return eld.detect(text);
}

// Funktion, mit der die Daten von der Page gescraped werden.
async function extractReviewsFromPage(page) {
    // Funktion wird auf Page ausgeführt.
    return await page.evaluate(() => {

        // eigene kleine Funktion, um Username zu erhalten. -> sehr verschachtelt auf Seitenstruktur
        function getUsername(el) {
            if (!el) return '';
            for (const node of el.childNodes) {
                // Username in Textknoten enthalten. -> dieser rausgefiltert
                if (node.nodeType === Node.TEXT_NODE) {
                    // Arbeiten mit RegEx, um überflüssige Leerzeichen zu entfernen und den Text zu trimmen.
                    const txt = (node.textContent || '').replace(/\s+/g, ' ').trim();
                    if (txt) return txt;
                }
            }
        }

        const cards = Array.from(document.querySelectorAll('.review-card'));
        // Aufrufen einzelner Reviews
        return cards.map(card => {
            const header = card.querySelector('.review-card__header') || card.querySelector('a.review-card__header') || card;
            // Username aus Header extrahiert.
            const username = getUsername(header) || '';
            const ratingText = card.querySelector('.c-siteReviewHeader_reviewScore, .c-siteReviewScore, .c-siteReviewScore_background')?.textContent || '';
            // Rating extrahiert.
            const rating = parseInt((ratingText.match(/\d+/) || ['0'])[0], 10) || 0;
            // Datum extrahiert.
            const date = card.querySelector('.review-card__date, .c-siteReview_reviewDate')?.textContent?.trim() || '';
            // Eigentlicher Reviewtext extrahiert.
            const review = card.querySelector('.review-card__quote, .c-siteReview_quote, .review-body, .review_body')?.textContent?.trim() || '';
            return {username, rating, date, review};
        });
    });
}

// Funktion, um Metadaten von Seiten der User-Profile zu sammeln.
async function getMetadata(userPage) {
    // Puppeteer Setup
    const browser = await puppeteer.launch();
    const page = await browser.newPage();
    await page.goto(userPage, {waitUntil: 'networkidle2'});

    // Extrahieren von durchschnittlichen User Score.
    const averageUserScore = await page.evaluate(() => {
        const scoreText = document.querySelector('.c-scoreOverview_avgScoreText')?.textContent?.trim() || '';
        return parseFloat(scoreText) || null;
    });

    // Extrahieren von Anzahl der bewerteten Spiele.
    const games = await page.evaluate(() => {
        const gameText = document.querySelector('.c-globalHeader_menu_subText');
        return gameText ? gameText.textContent.trim() : '';
    });

    // Extrahieren von der Bewertungsverteilung. -> Anzahl positiv, neutral und negativ bewerteter Spiele.
    const scoreCountTexts = await page.evaluate(() =>
        Array.from(document.querySelectorAll('.c-scoreCount_count')).map(n => (n?.textContent || '').trim())
    );

    // Normalisieren von den extrahierten Texten, um die Anzahl der Bewertungen zu erhalten. -> Entfernen von nicht-numerischen Zeichen und Umwandeln in Integer.
    const scoreCount = scoreCountTexts.map(t => {
        const normalized = t.replace(`^\d+`, '');
        return normalized ? parseInt(normalized, 10) : null;
    });
    //
    const scoreCounts = {
        positive: scoreCount[0] ?? 0,
        neutral: scoreCount[1] ?? 0,
        negative: scoreCount[2] ?? 0,
    };

    // Schließen des Browsers und Rückgabe der gesammelten Metadaten.
    await browser.close();
    return {averageUserScore, games, scoreCounts};
}

// Hilfsfunktion, um zum Ende der Seite zu scrollen, damit weitere Reviews geladen werden können.
async function scrollToBottom(page) {
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
}

// Hauptfunktion, mit der die Reviews gesammelt werden.
async function collectReviews(metadata = true) {
    const browser = await puppeteer.launch();
    const page = await browser.newPage();
    await page.goto(URL, {waitUntil: 'domcontentloaded'});

    // Initialisieren von Variablen. -> unter anderem das Array collected
    const collected = [];
    let negativeReviews = 0;
    let neutralReviews = 0;
    let positiveReviews = 0;

    // Abbruchbedingung: Solange nicht genügend Reviews in jeder Kategorie gesammelt wurden, wird weiter gescrollt und neue Reviews extrahiert.
    while (!(negativeReviews >= limit && neutralReviews >= limit && positiveReviews >= limit)) {

        // Reviews von der aktuellen Seite extrahieren.
        const reviews = await extractReviewsFromPage(page);

        // einzelnes Review wird rausgezogen.
        for (const review of reviews) {

            // Sprache wird erkannt und falls diese Englisch ist, wird es weiterverarbeitet.
            const lang = await detectLang(review.review || '');
            if (lang.language === 'en') {

                console.log(`Überprüfe Review von ${review.username} (Rating: ${review.rating}, Sprache: ${lang.language})`);
                // Duplikate werden ausgeschlossen.
                if (collected.some(r => (r.username === review.username))) continue;

                // Reviews mit Spoilerblocker werden geskippt.
                if (review.review === '[SPOILER ALERT: This review contains spoilers.]') continue;

                // wenn Review ein Rating von unter 4 hat. -> negative Kategorie
                if (review.rating < 4) {
                    if (negativeReviews < limit) {
                        negativeReviews++;
                        // wenn Metadaten Parameter aktiviert -> Metadaten gesammelt
                        if (metadata === true) {
                            console.log(`Review von ${review.username}`);
                            let userPage = `https://www.metacritic.com/user/${review.username}`;
                            const metadata = await getMetadata(userPage);
                            // Review wird in Array gepusht. -> mit oder ohne Metadaten
                            collected.push({...review, metadata});
                        } else {
                            collected.push(review);
                        }
                    }
                    // wenn Review ein Rating von über 7 hat. -> positive Kategorie
                } else if (review.rating > 7) {
                    if (positiveReviews < limit) {
                        positiveReviews++;
                        if (metadata === true) {
                            console.log(`Review von ${review.username}`);
                            let userPage = `https://www.metacritic.com/user/${review.username}`;
                            const metadata = await getMetadata(userPage);
                            collected.push({...review, metadata});
                        } else {
                            collected.push(review);
                        }
                    }
                } else {
                    // Andernfalls wird es der neutralen Kategorie zugeordnet.
                    if (neutralReviews < limit) {
                        neutralReviews++;
                        if (metadata === true) {
                            console.log(`Review von ${review.username}`);
                            let userPage = `https://www.metacritic.com/user/${review.username}`;
                            const metadata = await getMetadata(userPage);
                            collected.push({...review, metadata});
                        } else {
                            collected.push(review);
                        }
                    }
                }

                // Loggen von aktuellem Stand der Reviews, um den Fortschritt zu verfolgen.
                console.log(`Aktueller Stand - Negative: ${negativeReviews}, Neutral: ${neutralReviews}, Positive: ${positiveReviews}`);
                // Abbruchbedingung: Wenn genügend Reviews in jeder Kategorie gesammelt wurden, wird die Schleife verlassen.
                if (negativeReviews >= limit && neutralReviews >= limit && positiveReviews >= limit) break;
            }
        }

        // Scrollen zum Ende der Seite, um weitere Reviews zu laden, und kurze Pause, um die Seite nicht zu überlasten.
        await scrollToBottom(page);

        await sleep(3000);
    }

    console.log('Gefundene Reviews:', collected.length);
    console.log('Negative:', negativeReviews, 'Neutral:', neutralReviews, 'Positive:', positiveReviews);
    await browser.close();

    // Speichern der Reviews in data Folder und passende Bennennung.
    const jsonFilePath = `../data/${games[gameID]}${metadata ? '_with_metadata' : ''}.json`;
    fs.writeFileSync(jsonFilePath, JSON.stringify(collected, null, 2));
    console.log(`Daten in ${jsonFilePath} gespeichert.`);
}

collectReviews(true);
