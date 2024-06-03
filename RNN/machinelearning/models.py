import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.w, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        dot_product = self.run(x)
        result = nn.as_scalar(dot_product)
        return 1 if result >= 0 else -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        convergence = False
        while not convergence:
            convergence = True
            for x, y in dataset.iterate_once(1):  
                prediction = self.get_prediction(x)
                actual = nn.as_scalar(y)
                if prediction != actual:
                    convergence = False
                    self.w.update(x, actual)  


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 25
        self.hidden_size = 50

        self.W1 = nn.Parameter(1, self.hidden_size)  
        self.b1 = nn.Parameter(1, self.hidden_size)  

        # output layer
        self.output_w = nn.Parameter(self.hidden_size, 1)
        self.output_b = nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        hidden_output = nn.ReLU(nn.AddBias(nn.Linear(x, self.W1), self.b1))
        return nn.AddBias(nn.Linear(hidden_output, self.output_w), self.output_b)
    
    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        prediction = self.run(x)
        return nn.SquareLoss(prediction, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        learning_rate = -0.2
        
        while True:
            for row_vect, label in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(row_vect, label)
                params = [self.W1, self.output_w, self.b1, self.output_b]
                gradients = nn.gradients(loss, params)

                learning_rate = min(-0.01, learning_rate)

                #for param, grad in zip(params, gradients):
                #    param.update(grad, learning_rate)

                self.W1.update(gradients[0], learning_rate)
                self.output_w.update(gradients[1], learning_rate)
                self.b1.update(gradients[2], learning_rate)
                self.output_b.update(gradients[3], learning_rate)

            learning_rate += .02
            loss = self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y))
            if nn.as_scalar(loss) < 0.008:
                return

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        # Parameters based on the given architecture
        batch_size = 784  # MNIST images are 28x28 pixels, flattened to a 784 vector
        hidden_size = 200  # Suggested hidden layer size
        output_size = 10  # There are 10 classes for the digits 0-9

        # Initialize weights and biases for the two linear layers
        self.W1 = nn.Parameter(batch_size, hidden_size)
        self.b1 = nn.Parameter(1, hidden_size)
        self.W2 = nn.Parameter(hidden_size, output_size)
        self.b2 = nn.Parameter(1, output_size)

        # Learning rate
        self.learning_rate = 0.5

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        # First linear layer with ReLU activation
        xW1 = nn.Linear(x, self.W1)
        xW1_b1 = nn.AddBias(xW1, self.b1)
        relu = nn.ReLU(xW1_b1)
        
        # Output layer (no ReLU activation here as specified)
        reluW2 = nn.Linear(relu, self.W2)
        output = nn.AddBias(reluW2, self.b2)
        return output

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        scores = self.run(x)
        return nn.SoftmaxLoss(scores, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:
            total_loss = 0
            for x, y in dataset.iterate_once(100):  # Batch size of 100
                loss = self.get_loss(x, y)
                gradients = nn.gradients(loss, [self.W1, self.b1, self.W2, self.b2])
                
                # Update parameters
                self.W1.update(gradients[0], -self.learning_rate)
                self.b1.update(gradients[1], -self.learning_rate)
                self.W2.update(gradients[2], -self.learning_rate)
                self.b2.update(gradients[3], -self.learning_rate)

                # Optionally, convert loss to scalar and accumulate
                loss_scalar = nn.as_scalar(loss)
                total_loss += loss_scalar

            # Check validation accuracy after each epoch
            validation_accuracy = dataset.get_validation_accuracy()
            print(f"Validation Accuracy: {validation_accuracy}%")

            # Stopping condition based on validation accuracy
            if validation_accuracy > 0.98:  # Threshold set higher as per the discussion
                print("Satisfactory validation accuracy achieved. Stopping training.")
                break


class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 25       
        self.hidden_size = 350     

        self.w = nn.Parameter(self.num_chars, self.hidden_size)
        self.w_hidden = nn.Parameter(self.hidden_size, self.hidden_size)
        self.output_w = nn.Parameter(self.hidden_size, len(self.languages))
        self.initial_bias = nn.Parameter(1, self.hidden_size)
        self.hidden_bias = nn.Parameter(1, self.hidden_size)
    

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        hidden_state = nn.ReLU(nn.AddBias(nn.Linear(xs[0], self.w), self.initial_bias))

        for x in xs[1:]:
            combined_input = nn.Add(nn.Linear(x, self.w),
                                    nn.Linear(hidden_state, self.w_hidden))
            hidden_state = nn.ReLU(nn.AddBias(combined_input, self.hidden_bias))

        # output layer
        output = nn.Linear(hidden_state, self.output_w)
        return output

    def get_loss(self, xs, ys):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        prediction = self.run(xs)
        return nn.SoftmaxLoss(prediction, ys)
    
    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        learning_rate = -0.09
        max_gradient_norm = 5  # max norm for gradient clipping

        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                gradients = nn.gradients(loss, [self.w, self.w_hidden, self.output_w])

                learning_rate = min(-0.004, learning_rate)

                self.w.update(gradients[0], learning_rate)
                self.w_hidden.update(gradients[1], learning_rate)
                self.output_w.update(gradients[2], learning_rate)

                # for param, grad in zip([self.w, self.w_hidden, self.output_w], gradients):
                #    clipped_grad = nn.clip_gradient(grad, max_gradient_norm)
                #    param.update(clipped_grad, learning_rate)

            learning_rate += 0.002
            if dataset.get_validation_accuracy() >= 0.89:
                return

    