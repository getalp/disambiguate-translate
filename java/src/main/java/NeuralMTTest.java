import getalp.wsd.common.utils.ArgumentParser;
import getalp.wsd.method.neural.NeuralDisambiguator;
import getalp.wsd.ufsac.core.Corpus;
import getalp.wsd.ufsac.core.Document;
import getalp.wsd.ufsac.core.Paragraph;
import getalp.wsd.ufsac.core.Sentence;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import com.google.common.math.DoubleMath;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class NeuralMTTest
{
    private String pythonPath;

    private String dataPath;

    private List<String> weights;

    private List<String> testCorpusPaths;

    private List<String> languagePair;

    private boolean lowercase;

    private boolean clearText;

    private int batchSize;

    private int beamSize;

    private void test(String[] args) throws Exception
    {
        ArgumentParser parser = new ArgumentParser();
        parser.addArgument("python_path");
        parser.addArgument("data_path");
        parser.addArgumentList("weights");
        parser.addArgumentList("corpus");
        parser.addArgumentList("lang");
        parser.addArgument("lowercase", "false");
        parser.addArgument("clear_text", "true");
        parser.addArgument("batch_size", "1");
        parser.addArgument("beam_size", "1");
        if (!parser.parse(args, true)) return;

        pythonPath = parser.getArgValue("python_path");
        dataPath = parser.getArgValue("data_path");
        weights = parser.getArgValueList("weights");
        testCorpusPaths = parser.getArgValueList("corpus");
        languagePair = parser.getArgValueList("lang");
        lowercase = parser.getArgValueBoolean("lowercase");
        clearText = parser.getArgValueBoolean("clear_text");
        batchSize = parser.getArgValueInteger("batch_size");
        beamSize = parser.getArgValueInteger("beam_size");

        System.out.println();
        System.out.println("------ Evaluate the score of an ensemble of models");
        System.out.println();

        evaluate_ensemble();

        System.out.println();
        System.out.println("------ Evaluate the scores of individual models");
        System.out.println();

        evaluate_mean_scores();
    }

    private void evaluate_ensemble() throws Exception
    {
        NeuralDisambiguator neuralDisambiguator = new NeuralDisambiguator(pythonPath, dataPath, weights, clearText, batchSize, true, beamSize);
        neuralDisambiguator.lowercaseWords = lowercase;
        for (String testCorpusPath : testCorpusPaths)
        {
            System.out.println("Evaluate on corpus " + testCorpusPath);
            Corpus corpusSrc = loadCorpusFromTxt(testCorpusPath + "." + languagePair.get(0));
            Corpus corpusTgt = loadCorpusFromTxt(testCorpusPath + "." + languagePair.get(1));
            evaluate(neuralDisambiguator, corpusSrc, corpusTgt);
            System.out.println();
        }
        neuralDisambiguator.close();
    }

    private void evaluate_mean_scores() throws Exception
    {
        List<NeuralDisambiguator> neuralDisambiguators = new ArrayList<>();
        for (String weight : weights)
        {
            NeuralDisambiguator neuralDisambiguator = new NeuralDisambiguator(pythonPath, dataPath, weight, clearText, batchSize, true, beamSize);
            neuralDisambiguator.lowercaseWords = lowercase;
            neuralDisambiguators.add(neuralDisambiguator);
        }
        for (String testCorpusPath : testCorpusPaths)
        {
            System.out.println("Evaluate on corpus " + testCorpusPath);
            List<Double> results = new ArrayList<>();
            for (int i = 0; i < weights.size(); i++)
            {
                NeuralDisambiguator neuralDisambiguator = neuralDisambiguators.get(i);
                Corpus corpusSrc = loadCorpusFromTxt(testCorpusPath + "." + languagePair.get(0));
                Corpus corpusTgt = loadCorpusFromTxt(testCorpusPath + "." + languagePair.get(1));
                double result = evaluate(neuralDisambiguator, corpusSrc, corpusTgt);
                results.add(result);
            }
            System.out.println();
            System.out.println("Mean of scores without backoff: " + mean(results));
            System.out.println("Standard deviation without backoff: " + standardDeviation(results));
            System.out.println();
        }
        for (int i = 0; i < weights.size(); i++)
        {
            neuralDisambiguators.get(i).close();
        }
    }

    private Corpus loadCorpusFromTxt(String txtCorpusPath) throws Exception
    {
        BufferedReader reader = Files.newBufferedReader(Paths.get(txtCorpusPath));
        Corpus corpus = new Corpus();
        Document document = new Document(corpus);
        Paragraph paragraph = new Paragraph(document);
        reader.lines().forEach(line -> new Sentence(line, paragraph));
        return corpus;
    }

    private double mean(List<Double> scores)
    {
        return DoubleMath.mean(scores);
    }

    private double standardDeviation(List<Double> scores)
    {
        return new StandardDeviation().evaluate(scores.stream().mapToDouble(Double::doubleValue).toArray(), mean(scores));
    }

    private double evaluate(NeuralDisambiguator disambiguator, Corpus corpusSrc, Corpus corpusTgt) throws Exception
    {
        List<Sentence> srcSentences = corpusSrc.getSentences();
        List<Sentence> tgtSentences = corpusTgt.getSentences();
        List<Sentence> hypSentences = disambiguator.disambiguateAndTranslateDynamicSentenceBatch(srcSentences);
        assert(srcSentences.size() == tgtSentences.size());
        assert(srcSentences.size() == hypSentences.size());
        writeSentencesToFile(tgtSentences, ".bleutest_ref");
        writeSentencesToFile(hypSentences, ".bleutest_hyp");
        ProcessBuilder pb = new ProcessBuilder("sacrebleu", ".bleutest_ref");
        pb.redirectError(ProcessBuilder.Redirect.INHERIT);
        Process sacrebleuProcess = pb.start();
        BufferedReader sacrebleuProcessReader = new BufferedReader(new InputStreamReader(sacrebleuProcess.getInputStream()));
        BufferedWriter sacrebleuProcessWriter = new BufferedWriter(new OutputStreamWriter(sacrebleuProcess.getOutputStream()));
        writeSentencesToBufferedWriter(hypSentences, sacrebleuProcessWriter);
        sacrebleuProcessReader.lines().forEach(System.out::println);
        return 0;
    }

    private static void writeSentencesToFile(List<Sentence> sentences, String filePath) throws Exception
    {
        writeSentencesToBufferedWriter(sentences, Files.newBufferedWriter(Paths.get(filePath)));
    }

    private static void writeSentencesToBufferedWriter(List<Sentence> sentences, BufferedWriter out) throws Exception
    {
        for (Sentence sentence : sentences)
        {
            out.write(sentence.toString());
            out.newLine();
        }
        out.close();
    }

    public static void main(String[] args) throws Exception
    {
        new NeuralMTTest().test(args);
    }
}
