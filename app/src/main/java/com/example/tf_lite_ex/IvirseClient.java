package com.example.tf_lite_ex;

import android.content.Context;
import android.os.ConditionVariable;

import org.tensorflow.lite.examples.model.ModelController;

import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Map;

public class IvirseClient {
    private ModelController modelController;
    private Context context;
    private static String TAG = "IVIRSE";
    private int local_epochs = 1;
//    private final ConditionVariable isTraining = new ConditionVariable();

    public IvirseClient(Context context) {
        this.modelController = new ModelController(context);
        this.modelController.loadData();
        this.context = context;
    }

    public float[][] getWeights() {
        return modelController.getWeights();
    }

    public float[][] fit(float[][] params, MainActivity activity) {
        modelController.updateWeights(params);
        modelController.startTraining(activity);
        return getWeights();
    }

}
