#include<bits/stdc++.h>
using namespace std;
#define br cout<<"\n";
#define ll long long
#define loop(n) for(int i=0; i<(n); i++)
#define fr(i,init, n) for(int i=(init); i<(n); i++)
#define revl(i,init) for(int i=(init-1); i>=0; i--)
#define pb push_back
#define all(v) v.begin(),v.end()
#define nl "\n"


#include <random>
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<double> dist(-1.0, 1.0);

double getRand(){
    return dist(gen);
}


const double LR=0.1;




enum Activation {
    NONE,
    SIGMOID,
    RELU,
    TANH
};

double activate(double z, Activation activation) {
    switch (activation) {

        case SIGMOID:
            return 1.0 / (1.0 + std::exp(-z));

        case RELU:
            return std::max(0.0, z);

        case TANH:
            return std::tanh(z);

        case NONE:
            return z;
    }

    return z;
}

double activationDerivative(double z, Activation activation) {
    switch (activation) {

        case SIGMOID: {
            double s = 1.0 / (1.0 + std::exp(-z));
            return s * (1.0 - s);
        }

        case RELU:
            return z > 0.0 ? 1.0 : 0.0;

        case TANH: {
            double t = std::tanh(z);
            return 1.0 - t * t;
        }

        case NONE:
            return 1.0;
    }

    return 1.0;
}

double lossDerivative(double yHat, double y) {
    return yHat - y;
}








class Neuron{
private:
    vector<double> W;
    double b;
    Activation activation;
    double z;      
    double a;       
    double delta;   


public:
    Neuron(size_t input_sz, Activation activation){
        W.resize(input_sz);
        for(size_t i=0; i<W.size(); i++) W[i]=getRand();
        b = getRand();
        this->activation = activation;
        z = 0.0;
        a = 0.0;
        delta = 0.0;
    }

    double forward(vector<double>&X){
        z = b;
        for(size_t i=0; i<W.size(); i++){
            z += W[i]*X[i];
        }
        a = activate(z, activation);
        return a;
    }


    void computeOutputDelta(double y){
        delta = lossDerivative(a, y) * activationDerivative(z, activation);
    }

    void computeHiddenDelta(double weightedDeltaSum){
        delta = weightedDeltaSum * activationDerivative(z, activation);
    }

    void updateWeights(vector<double>&input){
        for(size_t i=0; i<W.size(); i++){
            W[i] -= LR * delta * input[i];
        }
        b -= LR * delta;
    }

    double weightAt(size_t i) const { return W[i]; }
    double getDelta()         const { return delta; }
    double getOutput()        const { return a; }

    ~Neuron(){}

};


class NN{
private:
    vector<vector<Neuron>> model;

    vector<vector<double>> forwardAll(vector<double>&X){
        vector<vector<double>> activations;
        activations.push_back(X);

        for(size_t i=0; i<model.size(); i++){
            vector<double> output;
            output.reserve(model[i].size());
            for(size_t j=0; j<model[i].size(); j++){
                output.push_back(model[i][j].forward(activations.back()));
            }
            activations.push_back(output);
        }
        return activations;
    }


public:

    NN(vector<pair<size_t, Activation>>& layers, size_t inputSize){

        size_t prevSize = inputSize;

        for(size_t i=0; i<layers.size(); i++){

            vector<Neuron> layer;
            for(size_t j=0; j<layers[i].first; j++){
                Neuron n1(prevSize, layers[i].second);
                layer.push_back(n1);
            }
            prevSize = layers[i].first;
            model.push_back(layer);
        }
    }


    void train(vector<double>&X, int label){

        vector<vector<double>> activations = forwardAll(X);

        int L = (int)model.size();

        size_t outSize = model.back().size();
        vector<double> Y(outSize, 0.0);
        if(label >= 0 && (size_t)label < outSize) Y[label] = 1.0;


        for(size_t j=0; j<model[L-1].size(); j++){
            model[L-1][j].computeOutputDelta(Y[j]);
        }

        for(int i=L-2; i>=0; i--){
            for(size_t j=0; j<model[i].size(); j++){

                double weightedDeltaSum = 0.0;
                for(size_t k=0; k<model[i+1].size(); k++){
                    weightedDeltaSum += model[i+1][k].weightAt(j) * model[i+1][k].getDelta();
                }

                model[i][j].computeHiddenDelta(weightedDeltaSum);
            }
        }

        for(int i=0; i<L; i++){
            for(size_t j=0; j<model[i].size(); j++){
                model[i][j].updateWeights(activations[i]);
            }
        }
    }


    vector<double> forward(vector<double>&X){
        return forwardAll(X).back();
    }


    int predict(vector<double>&X){
        vector<double> out = forward(X);
        int best = 0;
        for(size_t j=1; j<out.size(); j++){
            if(out[j] > out[best]) best = (int)j;
        }
        return best;
    }

};

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////


// ============================================================
// MNIST DATA LOADER
// ============================================================

uint32_t readInt(ifstream& file) {

    uint32_t value;

    file.read(
        reinterpret_cast<char*>(&value),
        sizeof(value)
    );

    // MNIST uses big-endian integers.
    value = ((value & 0x000000FF) << 24) |
            ((value & 0x0000FF00) << 8)  |
            ((value & 0x00FF0000) >> 8)  |
            ((value & 0xFF000000) >> 24);

    return value;
}


vector<vector<double>> loadImages(const string& filename) {

    ifstream file(filename, ios::binary);

    if (!file) {
        throw runtime_error(
            "Could not open image file: " + filename
        );
    }

    uint32_t magic = readInt(file);
    uint32_t count = readInt(file);
    uint32_t rows  = readInt(file);
    uint32_t cols  = readInt(file);

    if (magic != 2051) {
        throw runtime_error(
            "Invalid MNIST image file: " + filename
        );
    }

    vector<vector<double>> images(
        count,
        vector<double>(rows * cols)
    );

    for (uint32_t i = 0; i < count; i++) {

        for (uint32_t j = 0; j < rows * cols; j++) {

            unsigned char pixel;

            file.read(
                reinterpret_cast<char*>(&pixel),
                1
            );

            // 0-255 → 0-1
            images[i][j] = pixel / 255.0;
        }
    }

    return images;
}


vector<int> loadLabels(const string& filename) {

    ifstream file(filename, ios::binary);

    if (!file) {
        throw runtime_error(
            "Could not open label file: " + filename
        );
    }

    uint32_t magic = readInt(file);
    uint32_t count = readInt(file);

    if (magic != 2049) {
        throw runtime_error(
            "Invalid MNIST label file: " + filename
        );
    }

    vector<int> labels(count);

    for (uint32_t i = 0; i < count; i++) {

        unsigned char label;

        file.read(
            reinterpret_cast<char*>(&label),
            1
        );

        labels[i] = label;
    }

    return labels;
}


// ============================================================
// TEST MNIST
// ============================================================

void test2() {

    // 784 inputs
    // 128 hidden neurons
    // 10 output neurons

    vector<pair<size_t, Activation>> layers = {
        {128, RELU},
        {10, SIGMOID}
    };

    NN net(layers, 784);


    // --------------------------------------------------------
    // Load dataset
    // --------------------------------------------------------

    cout << "Loading MNIST...\n";

    vector<vector<double>> trainImages =
        loadImages("../data/train-images-idx3-ubyte");

    vector<int> trainLabels =
        loadLabels("../data/train-labels-idx1-ubyte");

    vector<vector<double>> testImages =
        loadImages("../data/t10k-images-idx3-ubyte");

    vector<int> testLabels =
        loadLabels("../data/t10k-labels-idx1-ubyte");


    cout << "Training images: "
         << trainImages.size() << "\n";

    cout << "Training labels: "
         << trainLabels.size() << "\n";

    cout << "Test images: "
         << testImages.size() << "\n";

    cout << "Test labels: "
         << testLabels.size() << "\n";


    // --------------------------------------------------------
    // Training
    // --------------------------------------------------------

    const int EPOCHS = 5;

    for (int epoch = 0; epoch < EPOCHS; epoch++) {

        int correct = 0;

        for (size_t i = 0; i < trainImages.size(); i++) {

            net.train(
                trainImages[i],
                trainLabels[i]
            );

            // Training accuracy
            if (net.predict(trainImages[i])
                == trainLabels[i]) {

                correct++;
            }

            // Progress
            if ((i + 1) % 1000 == 0) {

                cout << "\rEpoch "
                     << epoch + 1
                     << "/"
                     << EPOCHS
                     << " | "
                     << i + 1
                     << "/"
                     << trainImages.size()
                     << flush;
            }
        }

        cout << "\n";

        cout << "Training accuracy: "
             << 100.0 * correct / trainImages.size()
             << "%\n";
    }


    // --------------------------------------------------------
    // Testing
    // --------------------------------------------------------

    cout << "\nTesting...\n";

    int correct = 0;

    for (size_t i = 0; i < testImages.size(); i++) {

        int prediction =
            net.predict(testImages[i]);

        if (prediction == testLabels[i]) {
            correct++;
        }

        if ((i + 1) % 1000 == 0) {

            cout << "\rTesting: "
                 << i + 1
                 << "/"
                 << testImages.size()
                 << flush;
        }
    }

    cout << "\n";

    cout << "Test accuracy: "
         << 100.0 * correct / testImages.size()
         << "%\n";
}










void test(){

    vector<pair<size_t, Activation>> layers = {
        {4, TANH},
        {2, SIGMOID}
    };

    NN net(layers, 2);

    vector<vector<double>> Xs = {{0,0},{0,1},{1,0},{1,1}};
    vector<int>            Ys = { 0,    1,    1,    0    };

    for(int epoch=0; epoch<20000; epoch++){
        for(size_t s=0; s<Xs.size(); s++){
            net.train(Xs[s], Ys[s]);
        }
    }

    cout << "XOR results after training:\n";
    int correct = 0;
    for(size_t s=0; s<Xs.size(); s++){
        int pred = net.predict(Xs[s]);
        correct += (pred == Ys[s]);
        cout << (int)Xs[s][0] << " ^ " << (int)Xs[s][1]
             << " = " << pred << "  (expected " << Ys[s] << ")\n";
    }
    cout << "accuracy: " << correct << "/" << Xs.size() << "\n";
}




int main(){
    test2();
    return 0;
}
