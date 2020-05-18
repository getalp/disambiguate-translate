import getalp.wsd.common.utils.ArgumentParser;
import getalp.wsd.common.utils.RegExp;
import getalp.wsd.common.wordnet.WordnetHelper;
import getalp.wsd.method.Disambiguator;
import getalp.wsd.method.FirstSenseDisambiguator;
import getalp.wsd.method.neural.NeuralDisambiguator;
import getalp.wsd.ufsac.core.Sentence;
import getalp.wsd.ufsac.core.Word;
import getalp.wsd.ufsac.utils.CorpusPOSTaggerAndLemmatizer;
import getalp.wsd.utils.WordnetUtils;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

public class NeuralMTDecode
{
    private NeuralDisambiguator neuralDisambiguator;

    private BufferedWriter writer;

    private BufferedReader reader;

    private void decode(String[] args) throws Exception
    {
        ArgumentParser parser = new ArgumentParser();
        parser.addArgument("python_path");
        parser.addArgument("data_path");
        parser.addArgumentList("weights");
        parser.addArgument("lowercase", "false");
        parser.addArgument("sense_compression_synonyms", "true");
        parser.addArgument("sense_compression_hypernyms", "true");
        parser.addArgumentList("txt_corpus_features", Collections.singletonList("null"));
        parser.addArgument("add_wordkey_from_sensekey", "false");
        parser.addArgument("clear_text", "false");
        parser.addArgument("batch_size", "1");
        parser.addArgument("beam_size", "1");
        if (!parser.parse(args)) return;

        String pythonPath = parser.getArgValue("python_path");
        String dataPath = parser.getArgValue("data_path");
        List<String> weights = parser.getArgValueList("weights");
        boolean lowercase = parser.getArgValueBoolean("lowercase");
        boolean senseCompressionSynonyms = parser.getArgValueBoolean("sense_compression_synonyms");
        boolean senseCompressionHypernyms = parser.getArgValueBoolean("sense_compression_hypernyms");
        List<String> txtCorpusFeatures = parser.getArgValueList("txt_corpus_features");
        boolean addWordKeyFromSenseKey = parser.getArgValueBoolean("add_wordkey_from_sensekey");
        boolean clearText = parser.getArgValueBoolean("clear_text");
        int batchSize = parser.getArgValueInteger("batch_size");
        int beamSize = parser.getArgValueInteger("beam_size");

        Map<String, String> senseCompressionClusters = null;
        if (senseCompressionHypernyms)
        {
            senseCompressionClusters = WordnetUtils.getSenseCompressionClusters(WordnetHelper.wn30(), true, false, false);
        }

        if (txtCorpusFeatures.size() == 1 && txtCorpusFeatures.get(0).equals("null"))
        {
            txtCorpusFeatures = Collections.emptyList();
        }

        neuralDisambiguator = new NeuralDisambiguator(pythonPath, dataPath, weights, clearText, batchSize, true, beamSize);
        neuralDisambiguator.lowercaseWords = lowercase;
        neuralDisambiguator.reducedOutputVocabulary = senseCompressionClusters;
        neuralDisambiguator.filterLemma = false;

        reader = new BufferedReader(new InputStreamReader(System.in));
        writer = new BufferedWriter(new OutputStreamWriter(System.out));
        List<Sentence> sentences = new ArrayList<>();
        for (String line = reader.readLine(); line != null ; line = reader.readLine())
        {
            Sentence sentence = extractSentenceFromRawInput(line, txtCorpusFeatures, neuralDisambiguator.getInputFeatures(), neuralDisambiguator.getInputAnnotationNames(), senseCompressionSynonyms, senseCompressionClusters, addWordKeyFromSenseKey);
            sentences.add(sentence);
            if (sentences.size() >= batchSize)
            {
                decodeSentenceBatch(sentences);
                sentences.clear();
            }
        }
        decodeSentenceBatch(sentences);
        writer.close();
        reader.close();
        neuralDisambiguator.close();
    }

    private Sentence extractSentenceFromRawInput(String line, List<String> txtCorpusFeatures, int inputFeatures, List<String> inputAnnotationNames, boolean reduceInputSenseToSynset, Map<String, String> senseCompressionClusters, boolean addWordKeyFromSenseKey)
    {
        Sentence sentence = new Sentence();

        if (txtCorpusFeatures.isEmpty())
        {
            txtCorpusFeatures = new ArrayList<>();
            for (int i = 0; i < inputFeatures; i++)
            {
                txtCorpusFeatures.add(inputAnnotationNames.get(i));
            }
        }

        String[] words = line.split(RegExp.anyWhiteSpaceGrouped.pattern());
        for (String word : words)
        {
            Word ufsacWord = new Word();
            String[] wordFeatures = word.split(Pattern.quote("|"));
            if (wordFeatures.length < 1)
            {
                wordFeatures = new String[]{"/"};
            }
            ufsacWord.setValue(wordFeatures[0]);
            for (int i = 1; i < txtCorpusFeatures.size(); i++)
            {
                if (wordFeatures.length > i)
                {
                    ufsacWord.setAnnotation(txtCorpusFeatures.get(i), wordFeatures[i]);
                }
            }
            if (ufsacWord.hasAnnotation("wn30_key"))
            {
                if (addWordKeyFromSenseKey)
                {
                    String senseKey = ufsacWord.getAnnotationValue("wn30_key");
                    String lemma = WordnetUtils.extractLemmaFromSenseKey(senseKey);
                    String pos = WordnetUtils.extractPOSFromSenseKey(senseKey);
                    String wordKey = lemma + "%" + pos;
                    ufsacWord.setAnnotation("word_key", wordKey);
                }
                if (reduceInputSenseToSynset || senseCompressionClusters != null)
                {
                    String synsetKey = WordnetHelper.wn30().getSynsetKeyFromSenseKey(ufsacWord.getAnnotationValue("wn30_key"));
                    if (senseCompressionClusters != null)
                    {
                        synsetKey = senseCompressionClusters.get(synsetKey);
                    }
                    ufsacWord.setAnnotation("wn30_key", synsetKey);
                }
            }
            sentence.addWord(ufsacWord);
        }
        return sentence;
    }

    private void decodeSentenceBatch(List<Sentence> sentences) throws IOException
    {
        List<Sentence> outputSentences = neuralDisambiguator.disambiguateAndTranslateDynamicSentenceBatch(sentences);
        for (Sentence sentence : outputSentences)
        {
            writer.write(sentence.toString());
            writer.newLine();
        }
        writer.flush();
    }

    public static void main(String[] args) throws Exception
    {
        new NeuralMTDecode().decode(args);
    }
}

