# Naive Bayes in the Avocado Example

This project implements a very simple Naive Bayes classifier for the question:

> Given an avocado's color and softness, is it good to eat?

The code estimates probabilities from the training data and then applies Bayes' theorem to decide which class is more likely.

---

## 1. The basic idea

The main rule used here is Bayes' theorem:

$$
P(C \mid X) = \frac{P(X \mid C) \cdot P(C)}{P(X)}
$$

In simple words for this avocado example:

- C = target label, meaning the answer we want to predict: whether the avocado is eatable (`YES`) or not eatable (`NO`)
- X = the evidence we observe, such as `Color = BLACK` and `Softness = MUSHY`
- $P(C)$ = prior probability, meaning how likely the avocado is to be eatable or not eatable before we look at its features
- $P(X \mid C)$ = likelihood, meaning how likely this color and softness are if the avocado is eatable or not eatable
- $P(C \mid X)$ = posterior probability, meaning the chance of being eatable or not eatable after we see the avocado's features

This project is using a simple version of Bayes called Naive Bayes:

$$
P(C \mid X_1, X_2) \propto P(C) \cdot P(X_1 \mid C) \cdot P(X_2 \mid C)
$$

This means:

- first we decide whether the avocado is likely eatable (`YES`) or not eatable (`NO`)
- then we look at the color and softness as two separate clues
- we combine those clues to decide which result is more likely

This is called “naive” because it assumes the features are independent once the target result is known. In simple words, it assumes color and softness do not strongly affect each other after we already know whether the avocado is eatable or not.

---

## 2. What the code is learning from the training data

The data file contains rows of the form:

- color
- softness
- good_to_eat

Example:

```python
BLACK, MUSHY, YES
```

The code reads the CSV, converts strings like `BLACK`, `MUSHY`, and `YES` into enum values, and stores them as tuples.

```python
for line in reader:
    if len(line) > 0:
        color, softness, good_to_eat = line
        data.append((Color.value_of(...), Softness.value_of(...), GoodToEat.value_of(...)))
```

This turns the raw text into structured data the model can work with.

---

## 3. The training step: estimating the probabilities

The class `AvocadoPredictor` stores these three things:

```python
self.color_given_good_to_eat_pmf = defaultdict(lambda: defaultdict(float))
self.softness_given_good_to_eat_pmf = defaultdict(lambda: defaultdict(float))
self.good_to_eat_prior = defaultdict(float)
```

These are probability tables:

- `good_to_eat_prior[YES]` = probability that a random avocado is good to eat
- `color_given_good_to_eat_pmf[YES][BLACK]` = probability that color is black given the avocado is good to eat
- `softness_given_good_to_eat_pmf[YES][MUSHY]` = probability that softness is mushy given the avocado is good to eat

The training logic is here:

```python
def fit(self, data):
    good_to_eat_count = defaultdict(int)
    color_given_good_to_eat_counts = defaultdict(lambda: defaultdict(int))
    softness_given_good_to_eat_counts = defaultdict(lambda: defaultdict(int))

    for color, softness, good_to_eat in data:
        good_to_eat_count[good_to_eat] += 1
        color_given_good_to_eat_counts[good_to_eat][color] += 1
        softness_given_good_to_eat_counts[good_to_eat][softness] += 1
```

This counts how many rows belong to each class and how many of those rows have each color and softness.

Then the code converts counts into probabilities:

```python
total_count = sum(good_to_eat_count.values())
for good_to_eat_value, count in good_to_eat_count.items():
    self.good_to_eat_prior[good_to_eat_value] = count / total_count
```

This computes:

$$
P(\text{GoodToEat} = c) = \frac{\text{count of class } c}{\text{total rows}}
$$

Similarly:

```python
for good_to_eat_value, color_count in color_given_good_to_eat_counts.items():
    total_color = good_to_eat_count[good_to_eat_value]
    for color, count in color_count.items():
        self.color_given_good_to_eat_pmf[good_to_eat_value][color] = count / total_color
```

This computes:

$$
P(\text{Color} = color \mid \text{GoodToEat} = c) = \frac{\text{count of color in class } c}{\text{total samples in class } c}
$$

The same is done for softness:

$$
P(\text{Softness} = softness \mid \text{GoodToEat} = c) = \frac{\text{count of softness in class } c}{\text{total samples in class } c}
$$

This is exactly the statistical foundation of a Naive Bayes model: estimate each one-dimensional conditional probability from the training set.

---

## 4. Why these are called PMFs

PMF means probability mass function.

The code is essentially building a discrete probability table for each class:

- `good_to_eat_prior` is the PMF for the class label itself
- `color_given_good_to_eat_pmf` is a conditional PMF of color given class
- `softness_given_good_to_eat_pmf` is a conditional PMF of softness given class

For example, the code checks values like:

```python
assert(np.isclose(m.good_to_eat_prior[GoodToEat.YES], 0.52941))
assert(np.isclose(m.softness_given_good_to_eat_pmf[GoodToEat.YES][Softness.MUSHY], 0.111111))
```

These numbers mean:

- about 52.94% of all avocado samples are good to eat
- given the avocado is good to eat, the probability of `Softness = MUSHY` is about 11.11%

---

## 5. What Bayes is doing in this example

Suppose we want to classify a new avocado with:

- color = BLACK
- softness = MUSHY

We ask:

$$
P(\text{YES} \mid \text{BLACK, MUSHY})
$$

and

$$
P(\text{NO} \mid \text{BLACK, MUSHY})
$$

Using Bayes' theorem and the Naive Bayes assumption:

$$
P(\text{YES} \mid \text{BLACK, MUSHY}) \propto P(\text{YES}) \cdot P(\text{BLACK} \mid \text{YES}) \cdot P(\text{MUSHY} \mid \text{YES})
$$

$$
P(\text{NO} \mid \text{BLACK, MUSHY}) \propto P(\text{NO}) \cdot P(\text{BLACK} \mid \text{NO}) \cdot P(\text{MUSHY} \mid \text{NO})
$$

Then we compare the two values. The larger one is the predicted class.

This is exactly the theoretical logic behind the code.

---

## 6. Why the code is written like this

### 6.1 Prior probability

The prior tells us how common each class is before we look at the avocado features.

```python
self.good_to_eat_prior[good_to_eat_value] = count / total_count
```

This tells us the base rate of `YES` and `NO` in the dataset.

### 6.2 Likelihood

For each class, we estimate how likely each feature value is:

```python
self.color_given_good_to_eat_pmf[good_to_eat_value][color] = count / total_color
self.softness_given_good_to_eat_pmf[good_to_eat_value][softness] = count / total_softness
```

These values capture the class-conditional behavior.

### 6.3 Why the code multiplies the probabilities

The model assumes:

$$
P(\text{Color, Softness} \mid C) = P(\text{Color} \mid C) \cdot P(\text{Softness} \mid C)
$$

So for a given class, it multiplies the probability of the color and the probability of the softness.

This is why the code does a multiplication for the feature values when deciding which class is more likely.

Without this assumption, the model would need to learn the combined probability of color and softness together, which is much harder and needs much more data.

---

## 7. A small numerical example

From the training values in the code:

- `P(YES) = 0.52941`
- `P(NO) = 0.47059`
- `P(BLACK | YES) = 0.111111`
- `P(MUSHY | YES) = 0.111111`
- `P(BLACK | NO) = 0.375`
- `P(MUSHY | NO) = 0.375`

For feature pair `(BLACK, MUSHY)`:

$$
P(\text{YES} \mid \text{BLACK, MUSHY}) \propto 0.52941 \times 0.111111 \times 0.111111
$$

$$
P(\text{NO} \mid \text{BLACK, MUSHY}) \propto 0.47059 \times 0.375 \times 0.375
$$

The second value is much larger, so the model would classify this avocado as `NO`.

This shows the central Bayesian logic: the class with the higher posterior score wins.

---

## 8. Why this is a Bayes classifier, not just counting

A simple frequency table could count how often each class appears with each feature pair, but the Bayes approach is more general and statistically principled.

It does two things:

1. It estimates priors from the overall data distribution.
2. It uses class-conditional feature probabilities to update belief after seeing evidence.

That is exactly the Bayesian process:

- prior belief
- observe data
- compute posterior belief
- choose the most likely class

---

## 9. Connection to the theoretical model

The code is implementing the standard generative model of Naive Bayes:

$$
\hat{y} = \arg\max_c P(c) \prod_i P(x_i \mid c)
$$

Where:

- $c$ ranges over the class labels
- $x_i$ are the observed feature values
- the product combines the evidence from each feature

This is one of the simplest and most widely used probabilistic learning models.

It is especially effective when:

- the data are discrete or categorical,
- feature dimensions are modest,
- there are enough examples to estimate probabilities reliably.

---

## 10. Summary in simple language

This code is a basic Naive Bayes classifier for avocado quality.

It does this:

1. counts how many examples are in each class,
2. finds how common each class is in the data,
3. finds how often each color and softness appears in each class,
4. uses Bayes' rule to decide which class a new avocado belongs to.

So in simple words:

- prior = “How common is the class?”
- likelihood = “How well do the features match this class?”
- posterior = “After seeing the features, how likely is this class?”

That is the whole idea behind Bayesian classification.

---

## 11. Final takeaway

The key formula is:

$$
P(C \mid X) = \frac{P(X \mid C) \cdot P(C)}{P(X)}
$$

In plain English:

> We want to know the probability of the class after seeing the features. We do that by combining how common the class is with how well the observed features match that class.

This is exactly what the code is doing.
