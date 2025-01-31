from gensim import utils
from scipy import stats
import requests, openai
import numpy as np

def evaluate_openai_word_pairs(
        OPENAI_API_KEY, pairs, delimiter='\t', case_insensitive=True, dummy4unknown=False,
):
        """Compute correlation of the model with human similarity judgments. Explots out-of-word embeddings as per FastText algorithm
        
        Notes
        -----
        More datasets can be found at
        * http://technion.ac.il/~ira.leviant/MultilingualVSMdata.html
        * https://www.cl.cam.ac.uk/~fh295/simlex.html.

        Parameters
        ----------
        pairs : str
            Path to file, where lines are 3-tuples, each consisting of a word pair and a similarity value.
            See `test/test_data/wordsim353.tsv` as example.
        delimiter : str, optional
            Separator in `pairs` file.
        restrict_vocab : int, optional
            Ignore all 4-tuples containing a word not in the first `restrict_vocab` words.
            This may be meaningful if you've sorted the model vocabulary by descending frequency (which is standard
            in modern word embedding models).
        case_insensitive : bool, optional
            If True - convert all words to their uppercase form before evaluating the performance.
            Useful to handle case-mismatch between training tokens and words in the test set.
            In case of multiple case variants of a single word, the vector for the first occurrence
            (also the most frequent if vocabulary is sorted) is taken.
        dummy4unknown : bool, optional
            If True - produce zero accuracies for 4-tuples with out-of-vocabulary words.
            Otherwise, these tuples are skipped entirely and not used in the evaluation.

        Returns
        -------
        pearson : tuple of (float, float)
            Pearson correlation coefficient with 2-tailed p-value.
        spearman : tuple of (float, float)
            Spearman rank-order correlation coefficient between the similarities from the dataset and the
            similarities produced by the model itself, with 2-tailed p-value.
        oov_ratio : float
            The ratio of pairs with unknown words.

        """
        openai.api_key = OPENAI_API_KEY
        similarity_gold = []
        similarity_model = []
        oov = 0

        with utils.open(pairs, 'rb') as fin:
            for line_no, line in enumerate(fin):
                line = utils.to_unicode(line)
                if line.startswith('#'):
                    # May be a comment
                    continue
                else:
                    try:
                        if case_insensitive:
                            a, b, sim = [word.upper() for word in line.split(delimiter)]
                        else:
                            a, b, sim = [word for word in line.split(delimiter)]
                        sim = float(sim)
                    except (ValueError, TypeError):
                        #logger.info('Skipping invalid line #%d in %s', line_no, pairs)
                        continue
                    # get vectors
                    response = openai.Embedding.create(
                            input=a,
                            model="text-embedding-ada-002"
                    )
                    av = np.array(response['data'][0]['embedding'])
                    response = openai.Embedding.create(
                            input=b,
                            model="text-embedding-ada-002"
                    )
                    bv = np.array(response['data'][0]['embedding'])
                    similarity_gold.append(sim)  # Similarity from the dataset
                    # Similarity from the model
                    cos_sim = np.dot(av, bv.T) / (np.linalg.norm(av)*np.linalg.norm(bv))
                    similarity_model.append(cos_sim)
                    
        spearman = stats.spearmanr(similarity_gold, similarity_model)
        pearson = stats.pearsonr(similarity_gold, similarity_model)
        if dummy4unknown:
            oov_ratio = float(oov) / len(similarity_gold) * 100
        else:
            oov_ratio = float(oov) / (len(similarity_gold) + oov) * 100

        return pearson, spearman, oov_ratio

def evaluate_word_pairs(
        model, pairs, delimiter='\t', restrict_vocab=300000, case_insensitive=True, dummy4unknown=False,
):
        """Compute correlation of the model with human similarity judgments. Explots out-of-word embeddings as per FastText algorithm
        
        Notes
        -----
        More datasets can be found at
        * http://technion.ac.il/~ira.leviant/MultilingualVSMdata.html
        * https://www.cl.cam.ac.uk/~fh295/simlex.html.

        Parameters
        ----------
        pairs : str
            Path to file, where lines are 3-tuples, each consisting of a word pair and a similarity value.
            See `test/test_data/wordsim353.tsv` as example.
        delimiter : str, optional
            Separator in `pairs` file.
        restrict_vocab : int, optional
            Ignore all 4-tuples containing a word not in the first `restrict_vocab` words.
            This may be meaningful if you've sorted the model vocabulary by descending frequency (which is standard
            in modern word embedding models).
        case_insensitive : bool, optional
            If True - convert all words to their uppercase form before evaluating the performance.
            Useful to handle case-mismatch between training tokens and words in the test set.
            In case of multiple case variants of a single word, the vector for the first occurrence
            (also the most frequent if vocabulary is sorted) is taken.
        dummy4unknown : bool, optional
            If True - produce zero accuracies for 4-tuples with out-of-vocabulary words.
            Otherwise, these tuples are skipped entirely and not used in the evaluation.

        Returns
        -------
        pearson : tuple of (float, float)
            Pearson correlation coefficient with 2-tailed p-value.
        spearman : tuple of (float, float)
            Spearman rank-order correlation coefficient between the similarities from the dataset and the
            similarities produced by the model itself, with 2-tailed p-value.
        oov_ratio : float
            The ratio of pairs with unknown words.

        """
        ok_keys = model.index_to_key[:restrict_vocab]
        if case_insensitive:
            ok_vocab = {k.upper(): model.get_index(k) for k in reversed(ok_keys)}
        else:
            ok_vocab = {k: model.get_index(k) for k in reversed(ok_keys)}

        similarity_gold = []
        similarity_model = []
        oov = 0

        original_key_to_index = model.key_to_index
        model.key_to_index = ok_vocab

        with utils.open(pairs, 'rb') as fin:
            for line_no, line in enumerate(fin):
                line = utils.to_unicode(line)
                if line.startswith('#'):
                    # May be a comment
                    continue
                else:
                    try:
                        if case_insensitive:
                            a, b, sim = [word.upper() for word in line.split(delimiter)]
                        else:
                            a, b, sim = [word for word in line.split(delimiter)]
                        sim = float(sim)
                    except (ValueError, TypeError):
                        #logger.info('Skipping invalid line #%d in %s', line_no, pairs)
                        continue
                    # DT: decide if out of vocabulary by trying to access vector
                    try:
                        av=model[a]
                        bv=model[b]
                    except KeyError:
                        #if a not in ok_vocab or b not in ok_vocab:
                        oov += 1
                        if dummy4unknown:
                            #logger.debug('Zero similarity for line #%d with OOV words: %s', line_no, line.strip())
                            similarity_model.append(0.0)
                            similarity_gold.append(sim)
                            continue
                        else:
                            #logger.debug('Skipping line #%d with OOV words: %s', line_no, line.strip())
                            continue
                    similarity_gold.append(sim)  # Similarity from the dataset
                    similarity_model.append(model.similarity(a, b))  # Similarity from the model
        model.key_to_index = original_key_to_index
        spearman = stats.spearmanr(similarity_gold, similarity_model)
        pearson = stats.pearsonr(similarity_gold, similarity_model)
        if dummy4unknown:
            oov_ratio = float(oov) / len(similarity_gold) * 100
        else:
            oov_ratio = float(oov) / (len(similarity_gold) + oov) * 100

        #logger.debug('Pearson correlation coefficient against %s: %f with p-value %f', pairs, pearson[0], pearson[1])
        #logger.debug(
        #    'Spearman rank-order correlation coefficient against %s: %f with p-value %f',
        #    pairs, spearman[0], spearman[1]
        #)
        #logger.debug('Pairs with unknown words: %d', oov)
        model.log_evaluate_word_pairs(pearson, spearman, oov_ratio, pairs)
        return pearson, spearman, oov_ratio
