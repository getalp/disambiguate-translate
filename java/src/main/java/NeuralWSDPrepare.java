import java.util.*;

import getalp.wsd.common.wordnet.WordnetHelper;
import getalp.wsd.method.neural.NeuralDataPreparator;
import getalp.wsd.common.utils.ArgumentParser;
import getalp.wsd.utils.WordnetUtils;

public class NeuralWSDPrepare
{
    public static void main(String[] args) throws Exception
    {
        ArgumentParser parser = new ArgumentParser();

        parser.addArgument("data_path");
        parser.addArgumentList("train");
        parser.addArgumentList("dev", Collections.emptyList());
        parser.addArgument("dev_from_train", "0");
        parser.addArgument("corpus_format", "xml");
        parser.addArgumentList("txt_corpus_features", Collections.singletonList("null"));

        parser.addArgumentList("input_features", Collections.singletonList("surface_form"));
        parser.addArgumentList("input_embeddings", Collections.singletonList("null"));
        parser.addArgumentList("input_vocabulary", Collections.singletonList("null"));
        parser.addArgumentList("input_vocabulary_limit", Collections.singletonList("-1"));
        parser.addArgumentList("input_clear_text", Collections.singletonList("false"));

        parser.addArgumentList("output_features", Collections.singletonList("wn30_key"));
        parser.addArgumentList("output_vocabulary", Collections.singletonList("null"));
        parser.addArgumentList("output_feature_vocabulary_limit", Collections.singletonList("-1"));

        parser.addArgumentList("output_translations", Collections.emptyList());
        parser.addArgumentList("output_translation_features", Collections.singletonList("surface_form"));
        parser.addArgumentList("output_translation_vocabulary", Collections.singletonList("null"));
        parser.addArgumentList("output_translation_vocabulary_limit", Collections.singletonList("-1"));
        parser.addArgumentList("output_translation_clear_text", Collections.singletonList("false"));
        parser.addArgument("share_translation_vocabulary", "false");

        parser.addArgument("truncate_line_length", "80");
        parser.addArgument("exclude_line_length", "150");
        parser.addArgument("line_length_tokenizer", "null");
        parser.addArgument("lowercase", "false");
        parser.addArgument("uniform_dash", "false");
        parser.addArgument("sense_compression_hypernyms", "true");
        parser.addArgument("sense_compression_instance_hypernyms", "false");
        parser.addArgument("sense_compression_antonyms", "false");
        parser.addArgument("sense_compression_file", "");
        parser.addArgument("add_wordkey_from_sensekey", "false");
        parser.addArgument("add_monosemics", "false");
        parser.addArgument("remove_monosemics", "false");
        parser.addArgument("remove_duplicates", "true");

        if (!parser.parse(args, true)) return;

        String dataPath = parser.getArgValue("data_path");
        List<String> trainingCorpusPaths = parser.getArgValueList("train");
        List<String> devCorpusPaths = parser.getArgValueList("dev");
        int devFromTrain = parser.getArgValueInteger("dev_from_train");
        String corpusFormat = parser.getArgValue("corpus_format");
        List<String> txtCorpusFeatures = parser.getArgValueList("txt_corpus_features");

        List<String> inputFeatures = parser.getArgValueList("input_features");
        List<String> inputEmbeddings = parser.getArgValueList("input_embeddings");
        List<String> inputVocabulary = parser.getArgValueList("input_vocabulary");
        List<Integer> inputVocabularyLimits = parser.getArgValueIntegerList("input_vocabulary_limit");
        List<Boolean> inputClearText = parser.getArgValueBooleanList("input_clear_text");

        List<String> outputFeatures = parser.getArgValueList("output_features");
        List<String> outputVocabulary = parser.getArgValueList("output_vocabulary");
        List<Integer> outputFeatureVocabularyLimits = parser.getArgValueIntegerList("output_feature_vocabulary_limit");

        List<String> outputTranslations = parser.getArgValueList("output_translations");
        List<String> outputTranslationFeatures = parser.getArgValueList("output_translation_features");
        List<String> outputTranslationVocabulary = parser.getArgValueList("output_translation_vocabulary");
        List<Integer> outputTranslationVocabularyLimits = parser.getArgValueIntegerList("output_translation_vocabulary_limit");
        List<Boolean> outputTranslationClearText = parser.getArgValueBooleanList("output_translation_clear_text");
        boolean shareTranslationVocabulary = parser.getArgValueBoolean("share_translation_vocabulary");

        int maxLineLength = parser.getArgValueInteger("truncate_line_length");
        boolean lowercase = parser.getArgValueBoolean("lowercase");
        boolean uniformDash = parser.getArgValueBoolean("uniform_dash");
        boolean senseCompressionHypernyms = parser.getArgValueBoolean("sense_compression_hypernyms");
        boolean senseCompressionInstanceHypernyms = parser.getArgValueBoolean("sense_compression_instance_hypernyms");
        boolean senseCompressionAntonyms = parser.getArgValueBoolean("sense_compression_antonyms");
        String senseCompressionFile = parser.getArgValue("sense_compression_file");
        boolean addWordKeyFromSenseKey = parser.getArgValueBoolean("add_wordkey_from_sensekey");
        boolean addMonosemics = parser.getArgValueBoolean("add_monosemics");
        boolean removeMonosemics = parser.getArgValueBoolean("remove_monosemics");
        boolean removeDuplicateSentences = parser.getArgValueBoolean("remove_duplicates");

        Map<String, String> senseCompressionClusters = null;
        if (senseCompressionHypernyms || senseCompressionAntonyms)
        {
            senseCompressionClusters = WordnetUtils.getSenseCompressionClusters(WordnetHelper.wn30(), senseCompressionHypernyms, senseCompressionInstanceHypernyms, senseCompressionAntonyms);
        }
        if (!senseCompressionFile.isEmpty())
        {
            senseCompressionClusters = WordnetUtils.getSenseCompressionClustersFromFile(senseCompressionFile);
        }

        inputEmbeddings = padList(inputEmbeddings, inputFeatures.size(), "null");
        inputVocabulary = padList(inputVocabulary, inputFeatures.size(), "null");
        inputClearText = padList(inputClearText, inputFeatures.size(), false);
        inputVocabularyLimits = padList(inputVocabularyLimits, inputFeatures.size(), -1);

        outputVocabulary = padList(outputVocabulary, outputFeatures.size(), "null");
        outputFeatureVocabularyLimits = padList(outputFeatureVocabularyLimits, outputFeatures.size(), -1);

        outputTranslationVocabulary = padList(outputTranslationVocabulary, outputTranslationFeatures.size(), "null");
        outputTranslationVocabularyLimits = padList(outputTranslationVocabularyLimits, outputTranslationFeatures.size(), -1);
        outputTranslationClearText = padList(outputTranslationClearText, outputTranslationFeatures.size(), false);

        txtCorpusFeatures = replaceNullStringByNull(txtCorpusFeatures);
        inputEmbeddings = replaceNullStringByNull(inputEmbeddings);
        inputVocabulary = replaceNullStringByNull(inputVocabulary);
        outputFeatures = replaceNullStringByNull(outputFeatures);
        outputTranslationVocabulary = replaceNullStringByNull(outputTranslationVocabulary);

        txtCorpusFeatures = clearNullOnlyList(txtCorpusFeatures);
        outputFeatures = clearNullOnlyList(outputFeatures);

        NeuralDataPreparator preparator = new NeuralDataPreparator();

        preparator.setOutputDirectoryPath(dataPath);

        for (String corpusPath : trainingCorpusPaths)
        {
            preparator.addTrainingCorpus(corpusPath);
        }

        for (String corpusPath : devCorpusPaths)
        {
            preparator.addDevelopmentCorpus(corpusPath);
        }

        for (int i = 0; i < inputFeatures.size(); i++)
        {
            preparator.addInputFeature(inputFeatures.get(i), inputClearText.get(i), inputEmbeddings.get(i), inputVocabulary.get(i), inputVocabularyLimits.get(i));
        }

        for (int i = 0; i < outputFeatures.size(); i++)
        {
            preparator.addOutputFeature(outputFeatures.get(i), outputVocabulary.get(i), outputFeatureVocabularyLimits.get(i));
        }

        for (int i = 0; i < outputTranslations.size(); i++)
        {
            preparator.addOutputTranslation(outputTranslations.get(i), outputTranslationFeatures, outputTranslationClearText, outputTranslationVocabulary, outputTranslationVocabularyLimits);
        }

        preparator.setCorpusFormat(corpusFormat);
        preparator.setShareTranslationVocabulary(shareTranslationVocabulary);

        preparator.txtCorpusFeatures = txtCorpusFeatures;
        preparator.maxLineLength = maxLineLength;
        preparator.lowercaseWords = lowercase;
        preparator.uniformDash = uniformDash;
        preparator.multisenses = false;
        preparator.removeAllCoarseGrained = true;
        preparator.addMonosemics = addMonosemics;
        preparator.removeMonosemics = removeMonosemics;
        preparator.reducedOutputVocabulary = senseCompressionClusters;
        preparator.additionalDevFromTrainSize = devFromTrain;
        preparator.removeDuplicateSentences = removeDuplicateSentences;
        preparator.addWordKeyFromSenseKey = addWordKeyFromSenseKey;

        preparator.prepareTrainingFile();
    }

    private static <T> List<T> padList(List<T> list, int padSize, T padValue)
    {
        List<T> newList = new ArrayList<>(list);
        while (newList.size() < padSize)
        {
            newList.add(padValue);
        }
        return newList;
    }

    private static List<String> replaceNullStringByNull(List<String> list)
    {
        List<String> newList = new ArrayList<>(list);
        for (int i = 0 ; i < newList.size() ; i++)
        {
            if (newList.get(i).equals("null"))
            {
                newList.set(i, null);
            }
        }
        return newList;
    }

    private static List<String> clearNullOnlyList(List<String> list)
    {
        List<String> newList = new ArrayList<>(list);
        boolean onlyNull = true;
        for (String element : newList)
        {
            if (element != null)
            {
                onlyNull = false;
                break;
            }
        }
        if (onlyNull)
        {
            newList.clear();
        }
        return newList;
    }
}

