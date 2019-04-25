// Copyright (C) 2015  Julián Urbano <urbano.julian@gmail.com>
// Distributed under the terms of the MIT License.

package ti;


import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;

/**
 * A processor to extract terms from HTML documents.
 */
public class HtmlProcessor implements DocumentProcessor
{

	// P2
	HashSet<String> stopWords;
	/**
	 * Creates a new HTML processor.
	 *
	 * @param pathToStopWords the path to the file with stopwords, or {@code null} if stopwords are not filtered.
	 * @throws IOException if an error occurs while reading stopwords.
	 */
	public HtmlProcessor(File pathToStopWords) throws IOException
	{
		// P2
		// Load stopwords
		this.stopWords = new HashSet<String>();
		List<String> words = Files.readAllLines(pathToStopWords.toPath());
		this.stopWords.addAll(words);
	}

	/**
	 * {@inheritDoc}
	 */
	public Tuple<String, String> parse(String html)
	{
		// P2
		// Parse document
		Document doc;
		// Parse HTML text to object.
		doc = Jsoup.parse(html);
		// Get title
		String title = doc.title();
		// Get body
		Element body = doc.body();
		
		// Prevent NullPointerException in case title or body is empty
		String bodyString = "";
		if(title == null){
			title = "";
		} 
		if(body != null){
			bodyString = body.text();
		}
		
		//System.out.println();
		//System.out.println("title -->"+ title);
		//System.out.println("body --> "+ bodyString);
		return new Tuple<String, String>(title, bodyString); 

	}

	/**
	 * Process the given text (tokenize, normalize, filter stopwords and stemize) and return the list of terms to index.
	 *
	 * @param text the text to process.
	 * @return the list of index terms.
	 */
	public ArrayList<String> processText(String text)
	{
		ArrayList<String> terms = new ArrayList<>();

		// P2
		// Tokenizing, normalizing, stopwords, stemming, etc. 
		// Call our 4 steps of processing text.
		// Tokenize sentence by separating word from " " space and get result as array of string.
		ArrayList<String> tokens = tokenize(text);
		
		// For each term after tokenization, we do normalize (lower case) and check if it is stop word or not.
		// If it's a stopword, we skip (TODO confirm this)
		// 
		for(String term: tokens) {
			// Lower case
			String normalizedTerm = normalize(term);
			// If term exist in stop word, skip this word (TODO confirm)
			if(!isStopWord(normalizedTerm)) {
				// If term is not a stop word, then finding root word in stem method
				// then add term to result list.
				terms.add(stem(normalizedTerm));
			}
		}

		return terms;
	}

	/**
	 * Tokenize the given text.
	 *
	 * @param text the text to tokenize.
	 * @return the list of tokens.
	 */
	protected ArrayList<String> tokenize(String text)
	{
		ArrayList<String> tokens = new ArrayList<>();

		// P2
		// Split sentences to token by space
		List<String> listTokens = Arrays.asList(text.replaceAll("[^A-Za-z0-9']", " ").split("\\s+"));
		for(String token: listTokens) {
			tokens.add(token);
		}
		//System.out.println(tokens);
		
		return tokens;

	}

	/**
	 * Normalize the given term.
	 *
	 * @param text the term to normalize.
	 * @return the normalized term.
	 */
	protected String normalize(String text)
	{
		String normalized = null;

		// P2
		normalized = text.toLowerCase();
		return normalized;

	}

	/**
	 * Checks whether the given term is a stopword.
	 *
	 * @param term the term to check.
	 * @return {@code true} if the term is a stopword and {@code false} otherwise.
	 */
	protected boolean isStopWord(String term)
	{
		boolean isTopWord = false;

		// P2
		isTopWord = (stopWords.contains(term))? true : false;
		return isTopWord;
	}

	/**
	 * Stem the given term.
	 *
	 * @param term the term to stem.
	 * @return the stem of the term.
	 */
	protected String stem(String term)
	{
		String stem = null;

		// P2
		Stemmer stemer = new Stemmer();
		stemer.add(term.toCharArray(), term.length());
		stemer.stem();
		stem = stemer.toString();

		return stem;
	}
}
