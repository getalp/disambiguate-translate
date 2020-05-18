package getalp.wsd.method.result;

import java.util.ArrayList;
import java.util.List;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import com.google.common.math.DoubleMath;

public class MultipleDisambiguationResult
{
    private List<DisambiguationResult> results = new ArrayList<>();
    
    public MultipleDisambiguationResult()
    {
        
    }
    
    public void addDisambiguationResult(DisambiguationResult result)
    {
        results.add(result);
    }

    public double scoreMean()
    {
        try
        {
            return DoubleMath.mean(allScores());
        }
        catch (Exception e)
        {
            return Double.NaN;
        }
    }

    public double scoreMeanPerPOS(String pos)
    {
        try
        {
            return DoubleMath.mean(allScoresPerPOS(pos));
        }
        catch (Exception e)
        {
            return Double.NaN;
        }
    }

    public double scoreStandardDeviation()
    {
        try
        {
            return new StandardDeviation().evaluate(allScores(), scoreMean());
        }
        catch (Exception e)
        {
            return Double.NaN;
        }
    }

    public double scoreStandardDeviationPerPOS(String pos)
    {
        try
        {
            return new StandardDeviation().evaluate(allScoresPerPOS(pos), scoreMeanPerPOS(pos));
        }
        catch (Exception e)
        {
            return Double.NaN;
        }
    }

    public double timeMean()
    {
        try
        {
            return DoubleMath.mean(allTimes());
        }
        catch (Exception e)
        {
            return Double.NaN;
        }
    }

    public double[] allScores()
    {
        return results.stream().mapToDouble(DisambiguationResult::scoreF1).toArray();
    }

    public double[] allScoresPerPOS(String pos)
    {
        return results.stream().mapToDouble(res -> res.scoreF1PerPOS(pos)).toArray();
    }

    public double[] allTimes()
    {
        return results.stream().mapToDouble(res -> res.time).toArray();
    }
}