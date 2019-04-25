// Copyright (C) 2015  Julián Urbano <urbano.julian@gmail.com>
// Distributed under the terms of the MIT License.

package ti;

import java.util.*;

/**
 * Implements retrieval in a vector space with the cosine similarity function and a TFxIDF weight formulation.
 */
public class Cosine implements RetrievalModel
{
	public Cosine()
	{
		// empty
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public ArrayList<Tuple<Integer, Double>> runQuery(String queryText, Index index, DocumentProcessor docProcessor)
	{
		// P1
		// Extract the terms from the query  
		ArrayList<String> terms = docProcessor.processText(queryText);

		// Calculate the query vector
		ArrayList<Tuple<Integer, Double>> queryVector = computeVector(terms, index);
		// Calculate the document similarity
		ArrayList<Tuple<Integer, Double>> scores = computeScores(queryVector, index);
		return scores; // return results

	}

	/**
	 * Returns the list of documents in the specified index sorted by similarity with the specified query vector.
	 *
	 * @param queryVector the vector with query term weights.
	 * @param index       the index to search in.
	 * @return a list of {@link Tuple}s where the first item is the {@code docID} and the second one the similarity score.
	 */
	protected ArrayList<Tuple<Integer, Double>> computeScores(ArrayList<Tuple<Integer, Double>> queryVector, Index index)
	{
		ArrayList<Tuple<Integer, Double>> results = new ArrayList<>();

		Double sumWeightQuery = 0.0;
		// P1
		HashMap sims = new HashMap<Integer, Double>();
		int sizeQueryVector = queryVector.size();
		
		for(int  v=0; v < sizeQueryVector; v++) {
			Tuple<Integer, Double> term = queryVector.get(v);
			Integer termId = term.item1;
			// w_tq
			Double weightTermQuery = term.item2;
			
			sumWeightQuery = sumWeightQuery + (weightTermQuery*weightTermQuery);
			
			ArrayList<Tuple<Integer, Double>> termDocuments = index.invertedIndex.get(termId);
			for(Tuple<Integer, Double> termInDoc: termDocuments) {
				Integer docId = termInDoc.item1;
				// w_td
				Double weightTermDoc = termInDoc.item2;
				
				Double weight =  weightTermDoc * weightTermQuery;
				

				if(sims.containsKey(docId)) {
					Double prevWeight = (Double)sims.get(docId);
					Double sumWeight = prevWeight + weight;
					// Accumulate the previous result to sims[docId]
					sims.put(docId, sumWeight);
				}else {
					sims.put(docId, weight);
				}
			}
		}
		
		
		//sqrt the sumWeightQuery

		Double norm_Que = Math.sqrt(sumWeightQuery);

		
		Iterator it = sims.entrySet().iterator();
	    while (it.hasNext()) {
	        Map.Entry sim = (Map.Entry)it.next();
	        Integer docId = (Integer)sim.getKey();
	        Double similarity = (Double)sim.getValue();
	        Tuple<String, Double> doc = index.documents.get(docId);
	        Double docNorm = doc.item2;
	        
	        Double final_sim = similarity/(norm_Que*docNorm);
	        
	        results.add(new Tuple(docId, final_sim));
	        
	    }
		

		// Sort documents by similarity and return the ranking
		Collections.sort(results, new Comparator<Tuple<Integer, Double>>()
		{
			@Override
			public int compare(Tuple<Integer, Double> o1, Tuple<Integer, Double> o2)
			{
				return o2.item2.compareTo(o1.item2);
			}
		});
		return results;
	}

	/**
	 * Compute the vector of weights for the specified list of terms.
	 *
	 * @param terms the list of terms.
	 * @param index the index
	 * @return a list of {@code Tuple}s with the {@code termID} as first item and the weight as second one.
	 */
	protected ArrayList<Tuple<Integer, Double>> computeVector(ArrayList<String> terms, Index index)
	{
		ArrayList<Tuple<Integer, Double>> vector = new ArrayList<>();

		// P1
		HashSet<String> termSet = new HashSet<String>(terms);

		for (String term : termSet) {
			//Get term ID and iDF
			Tuple<Integer, Double> indexTerm = index.vocabulary.get(term);
			
			Integer termId = indexTerm.item1;
			Double iDF = indexTerm.item2;
			//System.out.println("Term ID: "+termId +" iDF: "+ iDF);
			
			//calculate tf from 1 + log(freq_ti)
			// from slide tf_ij = number of occurrence of term i in document j
			//(freq_ti)  => how many time t_i appears in the query

			
			// TODO confuse how to find the frequence of the term in the document
			// but professor said that "how many times t_i appear in the query"
			//TODO confirm the way to find tf
			
			//int freq = Collections.frequency(terms, term);
			//TODO confirm with processor if it's log based 2 or 10
//			double tf =  1 + Math.log(freq_ti);
			//double tf =  1 + Math.log(freq) / Math.log(2);
			// Compute weight and add posting
            double tf = 1.0 + Math.log(Collections.frequency(terms, term));
			
			//weight_ij = weight assigned to term i in document j
			//tf_ij number of occurrence of term i in document j
			
			
			//Since iDF is calculated so we don't have to calculate here (Math.log(N/ni))
			//- N = number of document in entire collection
			//- int N = index.documents.size();
			//- ni = number of documents with term i.
			//- int ni = index.invertedIndex.get(termId).size();
			
			Double weight = tf * iDF;
			// Add term ID and weight
			//System.out.println("Term ID: "+termId +", Weight: "+weight);
			vector.add(new Tuple(termId, weight));
			
		}

		return vector;
	}
}
