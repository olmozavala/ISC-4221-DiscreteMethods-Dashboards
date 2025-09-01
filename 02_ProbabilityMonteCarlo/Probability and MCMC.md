

# **A Student's Guide to Randomness and Computation**

## **Part 1: The Language of Chance – Fundamentals of Probability**

Welcome to the study of random processes in computation. At first glance, the worlds of computer science—built on deterministic logic and precise algorithms—and probability—the study of inherent uncertainty—may seem at odds. However, as this tutorial will demonstrate, the ability to model, generate, and analyze randomness is not just a niche topic but a cornerstone of modern computing. From making intelligent predictions in machine learning to securing global communications, probability provides the essential toolkit for building systems that can thrive in a complex and uncertain world. This first section lays the foundational language of probability, starting not with abstract axioms, but with the practical question: why does this matter?

### **1.1 Why Probability Matters in Computing**

Probability is the science of uncertainty, and in the real world, uncertainty is everywhere.1 For a computer program to perform a useful task, whether it's filtering spam emails, recommending a movie, or navigating a busy street, it must be able to reason about incomplete information and make smart guesses. Probability theory provides the formal framework for this kind of reasoning.

* **Machine Learning:** Many machine learning algorithms are fundamentally probabilistic. For example, the Naive Bayes Classifier, a popular algorithm for tasks like spam filtering and document classification, uses conditional probability to determine the likelihood that a given piece of data (like an email) belongs to a certain class (spam or not spam) based on its features (the words it contains).1 AI systems in self-driving cars use probabilistic models to predict the future actions of pedestrians and other vehicles, making decisions that maximize safety under uncertainty.1  
* **Randomized Algorithms:** Sometimes, the most efficient way to solve a problem is to introduce randomness into the algorithm itself. Algorithms like Randomized Quicksort use a random pivot selection to achieve excellent average-case performance, avoiding the worst-case scenarios that can plague deterministic versions.1 Monte Carlo and Las Vegas algorithms leverage randomness to find solutions to problems that are too complex to solve deterministically in a reasonable amount of time.1  
* **Network Performance and Systems:** In computer networks and operating systems, probability is essential for performance modeling. It is used to predict network delays, model traffic patterns, schedule jobs and resources efficiently, and provision server capacity to handle fluctuating demand.3  
* **Cryptography:** Modern security is built on a foundation of randomness. The generation of cryptographic keys, the creation of nonces (numbers used once) in secure protocols like TLS, and the salting of passwords all rely on unpredictable random numbers to be secure. If an adversary could predict these numbers, the entire security system would collapse.4

The journey to these advanced applications began with a simple game of chance. In the 17th century, mathematicians Blaise Pascal and Pierre de Fermat corresponded about how to fairly divide the stakes in an interrupted game.7 Their problem required a new way of thinking: not just about what had already happened, but about the likelihood of all possible future outcomes. To solve it, they enumerated every possible future sequence of events (coin flips) and counted how many led to a win for each player. This fundamental act—defining a space of possibilities and quantifying their likelihoods—is the conceptual origin of the very same reasoning that powers a modern AI. Whether the event is a coin flip or the next move of a pedestrian, the underlying challenge is to make a decision based on a formal model of an uncertain future. The principles developed by Pascal and Fermat provide the universal language for this task.

### **1.2 The Building Blocks: Sample Spaces, Events, and Random Variables**

To reason about uncertainty, we need a formal vocabulary. Let's build this vocabulary using the simple example of tossing two standard six-sided dice.

* **Random Variable:** A **random variable**, often denoted by a capital letter like X, is a variable whose value is a numerical outcome of a random phenomenon. In our example, we can define a random variable X to be the sum of the values of the two dice.7  
* **Sample Space:** The **sample space**, denoted by Ω (Omega), is the set of all possible outcomes for a random variable. For our random variable X, the smallest possible sum is 1+1=2 and the largest is 6+6=12. Therefore, the sample space is Ω={2,3,4,5,6,7,8,9,10,11,12}.7  
* **Event:** An **event**, denoted by a capital letter like E, is a subset of the sample space (E⊂Ω). It represents a specific outcome or set of outcomes that we are interested in. For instance, we might be interested in the event that the sum of the dice is even. In this case, the event E would be the set E={2,4,6,8,10,12}.7

The probability of an event, written as p(E), is a number between 0 (impossible) and 1 (certain) that measures its likelihood.7 The probability of an event is the sum of the probabilities of each individual outcome within that event. Two fundamental axioms govern probability:

1. The probability of any event E is 0≤p(E)≤1.  
2. The sum of the probabilities of all possible outcomes in the sample space is 1\. That is, p(Ω)=1.7

### **1.3 Combining Events: Unions, Intersections, and Complements**

Often, we are interested in how different events relate to each other. We can visualize these relationships using Venn diagrams.

* **Union (OR):** The union of two events, A∪B, is the event that *either* A *or* B (or both) occurs. For example, if we roll a single die, let event A be "the outcome is even" (A={2,4,6}) and event B be "the outcome is ≤3" (B={1,2,3}). The union is A∪B={1,2,3,4,6}.7  
* **Intersection (AND):** The intersection of two events, A∩B, is the event that *both* A *and* B occur. In the single die example, the intersection is A∩B={2}, as 2 is the only outcome that is both even and less than or equal to 3\.7

To calculate the probability of a union, we use the general addition rule:

p(A∪B)=p(A)+p(B)−p(A∩B)

This formula intuitively makes sense: we add the probabilities of A and B, but then we must subtract the probability of their intersection because we counted the outcomes in that overlap twice.7

* **Complement (NOT):** The **complement** of an event E, written as Ec, is the set of all outcomes in the sample space that are *not* in E. For example, if E is "3 or fewer hurricanes," Ec is "4 or more hurricanes".7 Since an event must either happen or not happen, their probabilities must sum to 1\. This gives us the  
  complement rule:

  p(Ec)=1−p(E)  
* **Mutually Exclusive Events:** Two events are **mutually exclusive** if they have no outcomes in common, meaning their intersection is empty (A∩B=∅). For example, when rolling a single die, the event "outcome is 1" and "outcome is 6" are mutually exclusive. In this case, p(A∩B)=0, and the addition rule simplifies to p(A∪B)=p(A)+p(B).7

### **1.4 Context is Everything: Conditional Probability and Independence**

The probability of an event can change if we are given additional information. This is the core idea behind conditional probability.

Imagine we are interested in the probability of two events: A={rain today} and B={rain tomorrow}. Intuitively, if we know that event A has occurred (it is raining today), our assessment of the probability of event B will likely increase, because weather patterns tend to persist.7 This means events

A and B are **dependent**.

The conditional probability of B given A, written as p(B∣A), is the probability that event B will occur, given that we know event A has already occurred. It is defined as:

p(B∣A)=p(A)p(A∩B)​

This can be understood as restricting our sample space. We are no longer considering all possible outcomes, only those in which A occurred. The formula then gives the fraction of those outcomes that also correspond to B occurring.7  
On the other hand, two events are independent if the occurrence of one does not affect the probability of the other. For example, the outcome of a coin flip today is independent of the outcome of a coin flip tomorrow. For independent events A and B, we have p(B∣A)=p(B). Substituting this into the definition of conditional probability and rearranging gives us the simple multiplication rule for independent events:

p(A∩B)=p(A)p(B)

This rule is a cornerstone of probability theory and is used extensively in analyzing systems with independent components.  
The following table summarizes these fundamental rules for easy reference.

| Rule Name | Formula | Plain Language Explanation |
| :---- | :---- | :---- |
| **Complement Rule** | p(Ac)=1−p(A) | The probability that an event *doesn't* happen is 1 minus the probability that it *does*. |
| **Addition Rule (General)** | p(A∪B)=p(A)+p(B)−p(A∩B) | The probability of A *or* B is the sum of their individual probabilities, minus the overlap (to avoid double-counting). |
| **Addition Rule (Mutually Exclusive)** | p(A∪B)=p(A)+p(B) | If A and B can't happen at the same time, the probability of A or B is simply the sum of their probabilities. |
| **Conditional Probability** | $p(B | A) \= \\frac{p(A \\cap B)}{p(A)}$ |
| **Multiplication Rule (General)** | $p(A \\cap B) \= p(A) \\cdot p(B | A)$ |
| **Multiplication Rule (Independent)** | p(A∩B)=p(A)⋅p(B) | If A and B are independent, the probability of both happening is the product of their individual probabilities. |

### **1.5 Worked Example – The Problem of Points**

Let's apply these concepts to solve the historical Pascal-Fermat game that started it all.

**The Problem:** Pascal and Fermat are playing a coin-flipping game. The first to 10 points wins a 100-franc pot. The game is interrupted when Fermat is winning 8 to 7\. How should the pot be divided fairly? 7

A fair division should be based on each player's probability of winning the game had it continued.

* **Step 1: Frame the Problem.**  
  * Fermat's score: 8\. He needs 10−8=2 more points to win.  
  * Pascal's score: 7\. He needs 10−7=3 more points to win.  
* Step 2: Determine the Sample Space.  
  The game must end within a certain number of additional coin flips. What is the maximum number of flips needed to guarantee a winner? Let's consider the worst-case scenario for a quick resolution. If the next 4 flips are, for example, H TTT (H for Fermat, T for Pascal), Fermat gets 1 point and Pascal gets 3 points. The score would become 9 to 10, and Pascal would win. In any sequence of 4 flips, one player is guaranteed to get enough points to win. For instance, if Fermat doesn't get 2 heads, Pascal must have gotten at least 3 tails. Therefore, we only need to consider the outcomes of the next 4 coin flips. The total number of possible outcomes is 24=16.7  
* Step 3: Calculate Probabilities for Each Outcome.  
  We assume a fair coin, so each of the 16 outcomes (HHHH, HHHT, HHTH, etc.) is equally likely, with a probability of 1/16. We now need to determine the winner for each of these 16 sequences. Let's analyze a few:  
  * **HHxx:** Fermat gets two heads in the first two flips. He wins immediately. The last two flips don't matter. There are 4 such outcomes: HHHH, HHHT, HHTH, HHTT.  
  * **HTHx:** Fermat gets his second head on the third flip. He wins. There are 2 such outcomes: HTHH, HTHT.  
  * **THHx:** Fermat gets his second head on the third flip. He wins. There are 2 such outcomes: THHH, THHT.  
  * **HTTH:** Fermat gets his second head on the fourth flip. He wins. This is 1 outcome.  
  * **THTH:** Fermat gets his second head on the fourth flip. He wins. This is 1 outcome.  
  * **TTHH:** Fermat gets his second head on the fourth flip. He wins. This is 1 outcome.  
  * **TTTx:** Pascal gets three tails in the first three flips. He wins immediately. There are 2 such outcomes: TTTT, TTTH.  
  * **TTHT:** Pascal gets his third tail on the fourth flip. He wins. This is 1 outcome.  
  * **THTT:** Pascal gets his third tail on the fourth flip. He wins. This is 1 outcome.  
  * **HTTT:** Pascal gets his third tail on the fourth flip. He wins. This is 1 outcome.  
* Step 4: Tally the Results.  
  Let's count the number of outcomes where each player wins:  
  * **Fermat wins:** 4 (from HHxx) \+ 2 (from HTHx) \+ 2 (from THHx) \+ 1 (HTTH) \+ 1 (THTH) \+ 1 (TTHH) \= 11 outcomes.  
  * Pascal wins: 2 (from TTTx) \+ 1 (TTHT) \+ 1 (THTT) \+ 1 (HTTT) \= 5 outcomes.  
    The probability of Fermat winning is p(Fermat wins)=11/16.  
    The probability of Pascal winning is p(Pascal wins)=5/16.  
    (Check: 11/16+5/16=16/16=1).7  
* Step 5: Conclude the Fair Division.  
  The 100-franc pot should be divided according to their respective probabilities of winning.  
  * Fermat's share: (11/16)×100=68.75 francs.  
  * Pascal's share: (5/16)×100=31.25 francs.

This problem beautifully illustrates how the abstract rules of probability can be used to arrive at a logical and fair solution to a complex problem involving uncertainty.

---

### **Self-Check Questions**

1. Think of an example from a computer program you have used or written. Can you identify a random variable, its sample space, and an interesting event?  
2. What is the key difference between mutually exclusive events and independent events? Can two events be both mutually exclusive and independent? (Hint: Consider their definitions in terms of intersection).

---

## **Part 2: Manufacturing Randomness – Random Numbers and Distributions**

Having established the language of probability, we now turn to a practical question: how can we generate the random outcomes needed for simulations and algorithms? This section explores the creation and description of random numbers on a computer, bridging the gap between the theoretical world of probability and the deterministic world of computation.

### **2.1 The Engine of Simulation: The Role of Random Numbers**

Random numbers are a fundamental resource in modern computing, serving as the fuel for a vast array of applications. Their importance stems from their ability to introduce unpredictability and to sample from large possibility spaces efficiently.

* **Cryptography:** As mentioned earlier, unpredictable random numbers are the lifeblood of security. They are used to generate secret keys for encryption, to create unique session identifiers, and to add "salt" to passwords before hashing them, which prevents attackers from using pre-computed tables (rainbow tables) to crack them. Without a source of high-quality randomness, cryptographic systems would be deterministic and, therefore, breakable.4  
* **Scientific Computing and Simulation:** Many complex physical and social systems are too difficult to model with exact analytical equations. Instead, scientists use random numbers to simulate these systems. This includes modeling the random motion of particles in a gas, the spread of a disease through a population, or the fluctuations in a financial market. The Monte Carlo method, which we will study in the next section, is a prime example of this approach.6  
* **Gaming and Procedural Generation:** Random numbers are essential in the gaming industry. They determine the outcome of events like a dice roll in a digital board game, the shuffling of cards in video poker, or the chance of finding a rare item in a role-playing game. They are also used in *procedural generation*, where algorithms create vast, unique game worlds, levels, or textures from an initial random seed, saving developers from having to design every detail by hand.6

### **2.2 The Two Faces of Randomness: True vs. Pseudo-Random**

A fundamental challenge arises from the nature of computers themselves. A traditional computer is a deterministic machine: if you provide it with the same input and run the same program, you will get the exact same output every single time. This is a feature, not a bug—it's what makes computation reliable. But it poses a problem for generating randomness: how can a perfectly predictable machine produce an unpredictable number?.8 This leads to a crucial distinction between two types of random number generators.

A **Pseudo-Random Number Generator (PRNG)** is a deterministic algorithm that takes an initial value, called a **seed**, and produces a sequence of numbers that appears random but is entirely determined by the seed. If you start with the same seed, you will get the exact same sequence of numbers. This is incredibly useful for debugging and scientific reproducibility, as you can re-run a simulation with the exact same "random" events.7 A simple example is the Linear Congruential Generator (LCG), defined by the recurrence relation

ni+1​=(a⋅ni​+c)modm, where a, c, and m are pre-chosen constants. The sequence is completely determined by the initial seed, n0​.7

A **True Random Number Generator (TRNG)**, in contrast, is a hardware device that generates randomness by tapping into an unpredictable physical process. These generators harvest *entropy* from the environment. Sources can include thermal noise from a resistor, atmospheric radio noise, the timing of radioactive decay, or even the tiny, unpredictable variations in a user's mouse movements or keystroke timings.5 Because the underlying physical processes are non-deterministic, a TRNG produces a sequence of numbers that is not reproducible and is considered truly random. Real-world examples include Intel's built-in

RDRAND hardware instruction and Cloudflare's famous "wall of lava lamps," which uses a camera to digitize the chaotic, unpredictable patterns of the lamps to generate seeds.5

The deterministic nature of a PRNG means that its entire, potentially infinite, sequence of outputs is completely predictable if the initial seed is known. This makes the seed the single most critical element for any security application. An attacker doesn't need to break the complex PRNG algorithm; they only need to guess or discover the seed to replicate the entire "random" sequence, which might include secret keys or other sensitive data.6 This vulnerability led to a famous security flaw in an early version of the Netscape web browser, which used a poorly seeded PRNG.

Because of this, modern secure systems employ a hybrid approach that leverages the strengths of both generator types. They use a slow but unpredictable TRNG to gather a small amount of high-quality physical entropy. This true randomness is then used to create a secure, unpredictable seed. This seed is then fed into a fast, efficient, and cryptographically secure PRNG (CSPRNG) to generate a large volume of pseudo-random numbers for applications. This strategy combines the unpredictability of the physical world with the speed of a deterministic algorithm, providing the best of both worlds.5

The following table summarizes the key differences between these two approaches.

| Feature | True Random Number Generator (TRNG) | Pseudo-Random Number Generator (PRNG) |
| :---- | :---- | :---- |
| **Source of Randomness** | Unpredictable physical processes (e.g., thermal noise, atmospheric noise, radioactive decay) | Deterministic mathematical algorithm |
| **Determinism** | Non-deterministic; the same process will not produce the same sequence twice. | Deterministic; given the same seed, it will always produce the same sequence. |
| **Seed** | Does not require a seed (it uses physical entropy as its input). | Requires an initial value called a seed to start the sequence. |
| **Speed** | Generally slower, as it's limited by the rate of physical entropy collection. | Very fast; can generate millions of numbers per second. |
| **Reproducibility** | Not reproducible. | Perfectly reproducible if the seed is known. |
| **Primary Use Case** | Generating seeds for PRNGs, high-security cryptography (e.g., one-time pads), scientific experiments requiring true randomness. | Simulations, scientific modeling, gaming, procedural generation, cryptography (when seeded by a TRNG). |

### **2.3 Describing Random Outcomes: Probability Distributions**

Once we have a way to generate random numbers, we need a mathematical tool to describe their behavior. A **Probability Distribution Function (PDF)** is a function that describes the relative likelihood for a random variable to take on a given value.7 The type of PDF depends on whether the random variable is discrete or continuous.

* **Discrete Probability Distributions:** For a discrete random variable, which can only take on a finite or countably infinite number of values, the PDF (more precisely called a Probability Mass Function or PMF) assigns a specific probability to each possible outcome. The sum of all these probabilities must equal 1\.7  
  * **Example: Discrete Uniform Distribution.** The roll of a single fair die is a classic example. The random variable X can take values in {1,2,3,4,5,6}. Since the die is fair, each outcome has an equal probability. The PDF is given by pX​(x)=1/6 for each x in the sample space.7  
* **Continuous Probability Distributions:** For a continuous random variable, which can take any value within a given range (e.g., height, temperature), the probability of it taking on any *single specific* value is zero. Instead, we talk about the probability of the variable falling within an interval. The PDF for a continuous variable is a curve where the **area under the curve** between two points gives the probability of the variable falling in that interval. The total area under the entire curve must be 1\.7  
  * **Example: Normal Distribution.** The Normal (or Gaussian) distribution, famous for its "bell curve" shape, is one of the most important distributions in statistics. It is described by two parameters: the mean (μ), which defines the center of the peak, and the standard deviation (σ), which defines the spread of the curve. Many natural phenomena, like human heights or measurement errors, can be approximated by a normal distribution.7

Another crucial function is the **Cumulative Distribution Function (CDF)**, denoted FX​(x). The CDF gives the probability that the random variable X will take a value *less than or equal to* x. That is, FX​(x)=p(X≤x).7

* For a discrete variable, the CDF is a step function that jumps up at each possible outcome.  
* For a continuous variable, the CDF is the integral of the PDF from −∞ to x, resulting in a smooth, S-shaped curve that goes from 0 to 1\.

### **2.4 Summarizing Distributions: Expected Value and Variance**

While a PDF or CDF gives a complete description of a random variable, it is often useful to summarize its properties with a few key numbers. The two most important are the expected value and the variance.

* **Expected Value (Mean):** The **expected value** of a random variable X, denoted E\[X\] or μ, represents its long-run average value. It is a weighted average of all possible outcomes, where the weights are the probabilities of those outcomes.7  
  * For a discrete random variable: E\[X\]=μ=∑x∈Ω​x⋅pX​(x)  
  * For a continuous random variable: E\[X\]=μ=∫−∞∞​x⋅pX​(x)dx  
* **Variance:** The **variance** of a random variable X, denoted Var(X) or σ2, measures the "spread" or dispersion of its distribution. It is defined as the expected value of the squared difference between the random variable and its mean.7  
  Var(X)=σ2=E\[(X−μ)2\]

  A small variance means the outcomes tend to be clustered closely around the mean, while a large variance means they are more spread out. A useful formula for computation is:

  Var(X)=E\[X2\]−(E\[X\])2

Example: Mean and Variance of a Fair Die Roll  
Let's calculate these values for our single fair die example, where pX​(x)=1/6 for x∈{1,2,3,4,5,6}.

1. Expected Value:  
   E\[X\]=(1⋅61​)+(2⋅61​)+(3⋅61​)+(4⋅61​)+(5⋅61​)+(6⋅61​)  
   E\[X\]=61​(1+2+3+4+5+6)=621​=3.5  
   The expected value is 3.5. Note that the expected value does not have to be a possible outcome itself.7  
2. Variance:  
   First, we calculate E\[X2\]:  
   E\[X2\]=(12⋅61​)+(22⋅61​)+(32⋅61​)+(42⋅61​)+(52⋅61​)+(62⋅61​)  
   E\[X2\]=61​(1+4+9+16+25+36)=691​  
   Now we use the computational formula for variance:  
   Var(X)=E\[X2\]−(E\[X\])2=691​−(3.5)2=691​−(27​)2=691​−449​  
   Var(X)=12182​−12147​=1235​≈2.9167  
   The standard deviation, σ, is the square root of the variance: σ=35/12​≈1.7078.

---

### **Self-Check Questions**

1. Why is reproducibility in a PRNG a desirable feature for scientific simulation but a critical flaw for generating a cryptographic key?  
2. The PDF for a continuous random variable (like the Normal distribution) can have a value greater than 1 at some points. Why does this not violate the axiom that probabilities must be between 0 and 1? (Hint: Think about area vs. height).

---

## **Part 3: Power in Numbers – The Monte Carlo Method**

With an understanding of probability and random number generation, we can now explore one of the most powerful and versatile computational techniques of the 20th century: the Monte Carlo method. This method uses randomness to solve problems that are often too complex for a direct analytical solution, providing approximate answers through statistical sampling.

### **3.1 Solving the Unsolvable by Sampling**

The Monte Carlo method was developed in the 1940s by scientists working on the Manhattan Project, including John von Neumann and Stanislaw Ulam.10 They needed to solve complex problems related to neutron diffusion, which were intractable with the mathematics of the day. The core idea they developed was to stop trying to solve the problem for every possibility and instead simulate the process using random numbers, observing the aggregate results. The method was named after the famous casino in Monaco, as it relies on the same principles of chance and repeated trials as games like roulette.10

Today, its applications are widespread and have a massive economic and scientific impact:

* **Finance and Business:** Financial analysts use Monte Carlo simulations to model the risk in investment portfolios, forecast stock prices, and price complex derivatives like options. By running thousands of simulations of potential market futures, each driven by random variables, they can build a distribution of possible outcomes and assess the probability of different profit or loss scenarios.12 Businesses use the same approach to model project profitability, forecast sales, and make strategic decisions under uncertainty.12  
* **Engineering:** Engineers use Monte Carlo methods to ensure the reliability and safety of their designs. For example, they can simulate the stress on a bridge under thousands of different randomly generated traffic and weather conditions to estimate its probability of failure. This is crucial in fields where building and testing physical prototypes is expensive or impossible.12  
* **Computer Graphics:** In the world of animated films and video games, a technique called path tracing, which is a form of Monte Carlo simulation, is used to create photorealistic images. The algorithm simulates the paths of millions of individual light rays as they bounce randomly around a virtual scene, accurately capturing complex lighting effects like soft shadows and color bleeding.14  
* **Physics and Science:** The method remains a vital tool in the physical sciences for simulating everything from the interactions of subatomic particles in a detector to the evolution of galaxies over billions of years.6

### **3.2 The Core Principle: Using Randomness to Estimate Quantities**

The fundamental idea behind the Monte Carlo method is to estimate a quantity by observing the frequency of a random event. The most classic example used to build intuition is the estimation of the value of π.

Imagine a square dartboard with side length 2, centered at the origin. Its area is 2×2=4. Now, inscribe a circle with radius 1 inside this square. The circle's area is πr2=π(1)2=π. The ratio of the circle's area to the square's area is Areasquare​Areacircle​​=4π​.7

Now, suppose we start throwing darts at this board completely at random, such that every point on the square is equally likely to be hit. After throwing a large number of darts, N, some will have landed inside the circle (Ninside​) and some outside. The ratio of darts inside the circle to the total number of darts should be approximately equal to the ratio of the areas:

$$\\frac{N\_{\\text{inside}}}{N} \\approx \\frac{\\text{Area}\_{\\text{circle}}}{\\text{Area}\_{\\text{square}}} \= \\frac{\\pi}{4}$$We can rearrange this to get an estimate for $\\pi$:$$\\pi \\approx 4 \\cdot \\frac{N\_{\\text{inside}}}{N}$$

The law of large numbers tells us that as we throw more and more darts (N→∞), this approximation will get closer and closer to the true value of π.  
Here is how we implement this algorithmically:

* **Step 1: Frame the Problem.** We want to estimate π. We set up a virtual 2x2 square spanning from (-1, \-1) to (1, 1\) and a circle of radius 1 centered at (0, 0).7  
* **Step 2: The "Dart Throwing" Simulation.** We generate N random points. For each point, we generate two random numbers, x and y, from a uniform distribution between \-1 and 1\.  
* **Step 3: The Test.** For each point (x,y), we check if it is inside the circle. The equation for a circle centered at the origin is x2+y2=r2. So, if x2+y2≤1, the point is inside the circle.7 We keep a counter,  
  Ninside​, and increment it every time this condition is met.  
* **Step 4: The Estimation.** After generating all N points, we compute our estimate using the formula πest​=4⋅(Ninside​/N).

This simple procedure demonstrates the core of the Monte Carlo method: replace a difficult deterministic problem (calculating the area of a circle) with a simple probabilistic one (checking if a random point is inside it) and repeat it many times.

### **3.3 Monte Carlo Integration and the Curse of Dimensionality**

The π-estimation example is a specific case of a more general technique called **Monte Carlo Integration**. The value of a definite integral, I=∫ab​f(x)dx, is equal to the average value of the function, f​, multiplied by the length of the integration interval, (b−a).7 The Monte Carlo method estimates this average value by sampling the function at

N random points xi​ in the interval and calculating their mean:

f​≈N1​i=1∑N​f(xi​)

The integral is then estimated as I≈(b−a)⋅N1​∑i=1N​f(xi​).  
For one-dimensional problems, this is often less efficient than deterministic methods like the trapezoidal rule. However, the true power of the Monte Carlo method becomes apparent in higher dimensions. This is due to a phenomenon known as the **Curse of Dimensionality**.7

Consider trying to integrate a function over a 10-dimensional hypercube. If we use a simple grid-based method and want to sample just 10 points along each dimension, we would need to evaluate the function at 1010 (ten billion) points. This is computationally infeasible. The number of points required for traditional methods grows exponentially with the number of dimensions, d.

The Monte Carlo method's convergence rate, however, is remarkably independent of dimension. The error of a Monte Carlo estimate typically decreases in proportion to 1/N​, where N is the number of samples, *regardless of the number of dimensions*. This means that for problems with many dimensions—common in fields like finance, physics, and machine learning—the Monte Carlo method is often the only viable approach.7

This ability to transform an uncertain model into a concrete distribution of possible outcomes is what makes the method a universal tool for risk quantification. A business doesn't just want a single forecast for next year's profit; they want to know the probability of making a loss, or the chances of exceeding their target. By modeling uncertain inputs (like sales and costs) as probability distributions and running a Monte Carlo simulation, they can generate a full probability distribution of potential profits. This allows for much richer, evidence-based decision-making, turning abstract uncertainty into quantifiable risk.12

### **3.4 The Theory Behind the Practice: The Central Limit Theorem**

Why can we be confident that the average of our random samples will converge to the correct answer? The theoretical guarantee comes from one of the most important theorems in all of statistics: the **Central Limit Theorem (CLT)**.

In simple terms, the Central Limit Theorem states that if you take a large number of independent and identically distributed random variables and calculate their average, the distribution of that average will be approximately a Normal (bell-shaped) distribution, *regardless of the original distribution of the variables*.7

The lecture notes provide an excellent illustration with dice rolls.7

* The distribution of a single die roll is uniform—each outcome from 1 to 6 is equally likely.  
* If we roll two dice and take their average, the distribution is no longer flat. It's now triangular, peaked at 3.5.  
* If we roll five, ten, or fifty dice and take their average, the distribution of that average looks more and more like a perfect bell curve.

This is precisely what happens in a Monte Carlo simulation. Each function evaluation at a random point, f(xi​), is a random variable. Our final estimate is the average of many of these random variables. The CLT tells us two crucial things:

1. The distribution of our estimate will be centered around the true value we are trying to find.  
2. The standard deviation of our estimate (which represents the error) will decrease in proportion to 1/N​. This confirms that our estimate gets more accurate as we increase the number of samples, and it gives us a precise formula for the rate of convergence.7

---

### **Self-Check Questions**

1. If you use the Monte Carlo method to estimate π and you want to reduce your error by a factor of 10, how many more samples do you need to run?  
2. Why is the Monte Carlo method particularly well-suited for problems involving many variables (high dimensionality), such as modeling the risk of a financial portfolio with hundreds of different stocks?

---

## **Part 4: Modeling Random Processes in Time**

In the previous sections, we focused on static probabilities and sampling. Now, we turn our attention to systems that evolve randomly over time. These are known as *stochastic processes*. We will explore two famous and influential examples: Brownian motion, which models continuous random walks, and the secretary problem, which models optimal decision-making in a sequence of random events.

### **4.1 Brownian Motion: The Random Walk of Particles and Prices**

In 1827, the botanist Robert Brown observed something peculiar while looking at pollen grains suspended in water through a microscope. The tiny grains were in a state of constant, erratic, jittery motion, even though the water was perfectly still. This phenomenon was named **Brownian motion**.7 For decades, it remained a mystery. In 1905, Albert Einstein published a groundbreaking paper explaining that this motion was direct evidence for the atomic theory. He theorized that the visible pollen grains were being constantly bombarded by a huge number of invisible, randomly moving water molecules. The cumulative effect of these tiny, independent collisions resulted in the random walk observed by Brown.16

The mathematical model for this process, also called a Wiener process, has two key properties derived from this physical intuition:

1. **Continuous Paths:** The particle moves continuously; it doesn't teleport from one point to another.  
2. **Independent, Normally Distributed Increments:** The displacement of the particle over any time interval is independent of its past motion and follows a Normal (Gaussian) distribution. The variance of this displacement is proportional to the length of the time interval.18

This means we can simulate a simple 2D random walk algorithmically:

1. Start a particle at the origin (x0​,y0​)=(0,0).  
2. For a chosen number of time steps, i=1,2,...,N:  
   * Generate two random numbers, dx and dy, from a Normal distribution with mean 0\.  
   * Update the particle's position: xi​=xi−1​+dx and yi​=yi−1​+dy.  
     This simple process generates the characteristic jagged, unpredictable paths of Brownian motion.7 A key result from Einstein's theory is that the average distance a particle travels from its starting point is proportional to the square root of the time elapsed,  
     t​.7

The fact that the same mathematical model can describe the physical motion of microscopic particles and the fluctuations of financial markets demonstrates the profound power of mathematical abstraction. A pollen grain's path is the cumulative result of countless tiny, random, independent collisions with water molecules. Similarly, a stock's price path can be seen as the cumulative result of countless buy and sell decisions made by traders, influenced by a mix of predictable trends and unpredictable news, sentiment, and random factors.19 In both cases, the system's state at the next moment is its current state plus a small, random increment. This is the essence of a random walk.

This insight led to the development of **Geometric Brownian Motion (GBM)**, a cornerstone of modern quantitative finance. GBM is used to model stock prices in the famous **Black-Scholes option pricing model**. It assumes that the stock price follows a random walk with two main components:

* A **drift** (μ), which represents the average expected return of the stock over time.  
* A **volatility** (σ), which represents the magnitude of the random fluctuations around that drift.16

  The mathematical tool used to describe this evolution is a Stochastic Differential Equation (SDE), which essentially says that the change in the stock price over a small time interval is composed of a deterministic part (the drift) and a random part driven by Brownian motion.16 This powerful abstraction allows tools and concepts developed in physics to be applied directly to the complex world of finance.

### **4.2 The Secretary Problem: Optimal Stopping and Decision-Making**

Let's shift from continuous processes to a discrete decision-making problem. Imagine you need to hire the best person for a job from a pool of N applicants. The rules are strict:

1. You interview applicants one by one in a random order.  
2. After each interview, you must immediately decide to either hire that person (ending the search) or reject them and move on.  
3. You cannot go back to an applicant you have already rejected.7

This is the classic **secretary problem**, a famous example of an *optimal stopping problem*. The core dilemma is the trade-off between "looking" and "leaping." If you hire too early, you might miss out on a much better candidate later. If you wait too long, the best candidate might have already been in the group you rejected, and you could be forced to hire someone mediocre at the end. This problem structure appears in many real-life scenarios: dating to find a partner, searching for an apartment, selling a house, or even deciding when to stop training a machine learning model to avoid overfitting.22

Remarkably, there is a mathematically optimal solution that maximizes your probability of selecting the single best candidate. The strategy is known as the **1/e rule** or the "look-then-leap" strategy:

**The Optimal Strategy:**

1. **Look Phase:** Reject the first N/e applicants, where e is Euler's number (≈2.718). This is approximately the first 37% of the applicants. During this phase, you do not hire anyone, but you pay close attention to the quality of the candidates to establish a benchmark. Let's call the score of the best candidate seen in this phase Q.7  
2. **Leap Phase:** After the first N/e applicants have been rejected, hire the *very next* applicant you interview whose score is greater than Q. If you reach the end of the applicant pool without finding such a candidate, you must hire the last person.7

This strategy gives you a probability of success (hiring the absolute best person) of approximately 1/e, or about 37%. This is a stunningly high probability, especially considering that a random guess would only give you a 1/N chance of success.21

While the 37% rule is mathematically optimal for the narrow goal of finding the single best candidate, its true value lies in the more general "explore-exploit" framework it illustrates. The rigid optimal solution has drawbacks: it fails to find anyone about 37% of the time (if the best candidate was in the initial "look" group), and it requires, on average, interviewing a large fraction of the candidates.7

What if we relax the goal? Instead of aiming for only the \#1 best candidate, what if we would be happy with hiring someone in the top 20%? The simulations in the lecture notes explore this exact question.7 They show that if you are willing to accept a "great" candidate instead of only the "best" one, the optimal strategy changes. The "look" phase becomes much shorter (around 10% of the pool), the risk of complete failure is reduced, and less effort is expended.

This reveals a more profound and practical lesson. The core of the problem is the **explore/exploit tradeoff**.25 The initial phase is

*exploration*—gathering data to understand the quality distribution and set a reasonable standard. The second phase is *exploitation*—committing to the first option that meets that standard. The optimal length of your exploration phase should depend on your specific goals and your tolerance for risk. This flexible framework for decision-making is a far more powerful takeaway than the single, rigid number of 37%.

---

### **Self-Check Questions**

1. Brownian motion describes a "memoryless" process. What does this mean in the context of modeling a stock price?  
2. In the secretary problem, what are the two main risks you are balancing when deciding how long the "look phase" should be?

---

## **Part 5: Synthesis and Practice**

This section consolidates the key concepts covered in this tutorial and provides a set of practice problems to test and deepen your understanding.

### **5.1 Key Takeaways**

* **Probability as a Language:** Probability is the formal language for reasoning under uncertainty. Core concepts like sample spaces, events, conditional probability, and independence are the building blocks for modeling complex systems in computer science.  
* **Randomness in Computation:** Computers use deterministic algorithms called **PRNGs** to generate sequences of numbers that appear random. These are fast and reproducible, making them ideal for simulation. For security, they must be seeded with unpredictable data from a hardware **TRNG**.  
* **Distributions as Descriptions:** The behavior of a random variable is described by its **probability distribution** (PDF/CDF). This distribution can be summarized by its **expected value** (mean) and **variance** (spread).  
* **Monte Carlo for Complexity:** The **Monte Carlo method** uses random sampling to find approximate solutions to complex problems, especially those in high dimensions where deterministic methods fail due to the **curse of dimensionality**. Its reliability is guaranteed by the **Central Limit Theorem**.  
* **Modeling Random Processes:** Stochastic processes model systems that evolve randomly over time. **Brownian motion** provides a model for continuous random walks used in physics and finance. The **secretary problem** provides a framework for optimal decision-making in sequential, uncertain scenarios, illustrating the fundamental **explore/exploit tradeoff**.

### **5.2 Practice Problems**

#### **Problem 1: Conceptual (Conditional Probability)**

A company has two servers that host its website. Server A handles 60% of the traffic and Server B handles the remaining 40%. The probability that Server A fails on any given day is 0.01. The probability that Server B fails is 0.02. Assuming the servers fail independently, what is the probability that the website is down on a given day? (The website is down if the server handling a user's request fails).

**Hint:** Use the law of total probability. The event "website is down" can be broken down into two mutually exclusive cases: (1) the user was routed to Server A AND Server A failed, or (2) the user was routed to Server B AND Server B failed.

Solution:  
Let A be the event that a user is routed to Server A, and B be the event for Server B.  
Let FA​ be the event that Server A fails, and FB​ be the event that Server B fails.  
We are given:  
p(A)=0.60  
p(B)=0.40  
p(FA​)=0.01  
p(FB​)=0.02  
The event "Website is Down" (D) can be expressed as (A∩FA​)∪(B∩FB​). Since a user is routed to either A or B, these two events are mutually exclusive. Therefore:  
p(D)=p(A∩FA​)+p(B∩FB​)  
Since the choice of server and its failure are independent events, we can use the multiplication rule for independent events:  
p(A∩FA​)=p(A)⋅p(FA​)=0.60⋅0.01=0.006  
p(B∩FB​)=p(B)⋅p(FB​)=0.40⋅0.02=0.008  
So, the total probability of the website being down is:  
p(D)=0.006+0.008=0.014  
There is a 1.4% chance the website is down for any given user request.

#### **Problem 2: Conceptual (RNGs)**

Explain a scenario in computer science where you would strongly prefer to use a PRNG over a TRNG, and a scenario where you would strongly prefer a TRNG over a PRNG. Justify your choices based on the properties of each generator type.

**Solution:**

* **Scenario for PRNG:** Scientific simulation or debugging a complex program. Suppose you are running a Monte Carlo simulation of a physical system. If you use a TRNG, the results will be different every time you run the simulation, making it impossible to debug or verify that a change in your code improved the result. By using a PRNG with a fixed seed, you guarantee that the sequence of "random" numbers is identical for each run. This reproducibility is essential for debugging, testing, and ensuring that scientific results can be verified by others.  
* **Scenario for TRNG:** Generating a master secret key for a Certificate Authority (CA) in a public key infrastructure. This key is the root of trust for a huge number of secure communications. If this key were generated by a PRNG and an attacker could somehow discover or guess the seed, they could regenerate the key and compromise the entire system. For a root-of-trust secret like this, you need the highest possible level of unpredictability, which can only be provided by a TRNG that harvests entropy from the physical world. The slower speed of a TRNG is irrelevant here, as this key is generated very rarely.

#### **Problem 3: Algorithm Tracing (Monte Carlo)**

Use the Monte Carlo method to estimate the value of the integral I=∫01​x2dx. The exact value is 1/3.  
Use N=5 sample points. Suppose your PRNG gives you the following five uniform random numbers between 0 and 1:  
u={0.8,0.2,0.5,0.9,0.1}  
Trace the steps of the algorithm and calculate the final estimate.

Solution:  
The function to be integrated is f(x)=x2. The integration interval is $$, so the length is (b−a)=1−0=1.  
The Monte Carlo estimate is given by I≈(b−a)⋅N1​∑i=1N​f(xi​).  
Here, N=5 and our random points xi​ are the values from u.

1. **Evaluate the function at each random point:**  
   * f(0.8)=0.82=0.64  
   * f(0.2)=0.22=0.04  
   * f(0.5)=0.52=0.25  
   * f(0.9)=0.92=0.81  
   * f(0.1)=0.12=0.01  
2. Sum the function values:  
   Sum \= 0.64+0.04+0.25+0.81+0.01=1.75  
3. Calculate the average value:  
   Average f​=NSum​=51.75​=0.35  
4. Calculate the integral estimate:  
   Iest​=(b−a)⋅f​=1⋅0.35=0.35

The Monte Carlo estimate with N=5 is 0.35. The true value is 1/3≈0.333. The estimate is close, and would get closer as N increases.

#### **Problem 4: Algorithm Tracing (Secretary Problem)**

You are using the secretary problem strategy to hire one of 10 applicants. Their true quality scores (which you only discover upon interview) arrive in the following random order:  
\`\`  
(10 is the best, 1 is the worst).  
You decide to use the optimal strategy. The number of applicants to "look" at is ⌊N/e⌋=⌊10/2.718⌋=⌊3.678⌋=3.  
Trace the algorithm. Which applicant do you hire? Did you succeed in hiring the best one?  
**Solution:**

1. **Set up:** N=10. The "look" phase size is 3\. The applicant scores arrive as \`\`.  
2. **Look Phase (Applicants 1 to 3):**  
   * **Interview Applicant 1 (Score 5):** This is the best score seen so far. Benchmark Q is set to 5\. Reject this applicant as per the rules.  
   * **Interview Applicant 2 (Score 8):** This score (8) is better than the current benchmark (5). Update the benchmark Q to 8\. Reject this applicant.  
   * **Interview Applicant 3 (Score 2):** This score (2) is not better than the benchmark (8). The benchmark remains Q=8. Reject this applicant.  
   * The "look" phase is now over. The final benchmark is Q=8.  
3. Leap Phase (Applicants 4 onwards):  
   The rule is now: hire the first applicant with a score greater than 8\.  
   * **Interview Applicant 4 (Score 10):** Is 10\>8? Yes.  
   * **Decision:** Hire Applicant 4\. The search stops.

**Conclusion:** You hire the applicant with score 10\. The best possible score was 10\. Yes, the strategy succeeded in this case.

#### **Problem 5: Short Proof**

Prove that if two events A and B are independent, then their complements, Ac and Bc, are also independent.

**Hint:** Start with the definition of independence for A and B: p(A∩B)=p(A)p(B). Use De Morgan's laws ($ (A \\cup B)^c \= A^c \\cap B^c $) and the addition rule.

Solution:  
We want to prove that p(Ac∩Bc)=p(Ac)p(Bc).

1. Start with the left-hand side, p(Ac∩Bc). By De Morgan's laws, we know that Ac∩Bc=(A∪B)c.  
   So, p(Ac∩Bc)=p((A∪B)c).  
2. Using the complement rule, p(Ec)=1−p(E), we can write:  
   p((A∪B)c)=1−p(A∪B).  
3. Now, apply the general addition rule for p(A∪B):  
   1−p(A∪B)=1−(p(A)+p(B)−p(A∩B)).  
   \=1−p(A)−p(B)+p(A∩B).  
4. We are given that A and B are independent, so we can substitute p(A∩B)=p(A)p(B):  
   \=1−p(A)−p(B)+p(A)p(B).  
5. Now, we factor this expression. This is the crucial algebraic step:  
   \=(1−p(A))−p(B)(1−p(A)).  
   \=(1−p(A))(1−p(B)).  
6. Finally, use the complement rule again: (1−p(A))=p(Ac) and (1−p(B))=p(Bc).  
   \=p(Ac)p(Bc).

We have shown that p(Ac∩Bc)=p(Ac)p(Bc), which is the definition of independence for Ac and Bc. Therefore, the complements are also independent.

### **5.3 Extension Questions**

1. **Monte Carlo Variance Reduction:** The efficiency of a Monte Carlo simulation is determined by its variance; a lower variance means fewer samples are needed for the same level of accuracy. Research one technique for *variance reduction*, such as "Importance Sampling" or "Stratified Sampling." Briefly describe how the technique works and provide an intuitive explanation for why it reduces the variance of the estimate compared to simple random sampling.  
2. **Secretary Problem Variants:** The classic secretary problem has rigid rules. Consider how the optimal strategy might change under the following modifications (you don't need a full mathematical proof, just a reasoned argument):  
   * **Variant A:** You are allowed to recall *one* previously rejected candidate at the very end of the process if you haven't hired anyone.  
   * **Variant B:** Your goal is not to maximize the probability of hiring the single best candidate, but to maximize the *expected score* of the candidate you hire.

## **Part 6: Continuing Your Journey**

This tutorial provides a solid foundation in the principles of randomness and computation. To continue building your intuition and exploring these topics in greater depth, the following external resources are highly recommended.

### **6.1 Recommended Videos**

1. **Khan Academy \- "Probability explained | Independent and dependent events"** 26  
   * **What it covers:** This video provides a clear, foundational explanation of the core concepts of probability, focusing on the crucial distinction between independent and dependent events using simple, visual examples like drawing marbles from a bag.  
   * **Why it's relevant:** It is an excellent resource for reinforcing the fundamental ideas from Part 1 of this tutorial. If you are still solidifying your understanding of conditional probability and the multiplication rule, this video's step-by-step approach is extremely helpful.  
2. **Acerola \- "Monte Carlo Simulation"** 14  
   * **What it covers:** This video gives a fantastic visual intuition for how Monte Carlo simulations work. It uses two compelling examples: estimating the value of π by randomly dropping marbles and simulating the flow of light (path tracing) to create realistic computer graphics.  
   * **Why it's relevant:** It perfectly complements Part 3 by showing, not just telling, how random sampling can solve complex problems. The computer graphics example is a powerful demonstration of the method's application in a modern, high-tech field.

### **6.2 Recommended Reading**

1. **Statistics by Jim \- "What is a Probability Distribution?"** 27  
   * **What it covers:** This blog post offers a very clear and accessible breakdown of the difference between discrete and continuous probability distributions. It provides numerous examples of common distributions (Binomial, Poisson, Normal, etc.) and explains the practical meaning of PMFs and PDFs.  
   * **Why it's relevant:** This article is an ideal supplement to Part 2\. It expands on the definitions of probability distributions with more examples and clear graphs, helping to build a stronger intuitive and practical understanding of how to describe random variables.  
2. **cdemi.io Blog \- "Maximizing the chances of finding the right one by solving the secretary problem"** 23  
   * **What it covers:** This article does an excellent job of translating the abstract secretary problem into a practical framework for real-life decision-making, using dating as its primary analogy. It discusses the famous 37% rule but also thoughtfully explores the limitations and real-world complexities that the simple model doesn't capture.  
   * **Why it's relevant:** It reinforces the key takeaway from Part 4: that the true lesson of the secretary problem is the "explore-exploit" framework, not just a single magic number. It encourages critical thinking about how to apply mathematical models to messy, real-world problems.

#### **Works cited**

1. Probability in Computer Science \- GeeksforGeeks, accessed August 15, 2025, [https://www.geeksforgeeks.org/maths/applications-of-probability/](https://www.geeksforgeeks.org/maths/applications-of-probability/)  
2. www.geeksforgeeks.org, accessed August 15, 2025, [https://www.geeksforgeeks.org/maths/applications-of-probability/\#:\~:text=Uses%20of%20Probability%20in%20Computer,conditional%20probability%20to%20classify%20data.](https://www.geeksforgeeks.org/maths/applications-of-probability/#:~:text=Uses%20of%20Probability%20in%20Computer,conditional%20probability%20to%20classify%20data.)  
3. Introduction to Probability for Computing, accessed August 15, 2025, [https://www.cs.cmu.edu/\~harchol/Probability/book.html](https://www.cs.cmu.edu/~harchol/Probability/book.html)  
4. en.wikipedia.org, accessed August 15, 2025, [https://en.wikipedia.org/wiki/Applications\_of\_randomness\#:\~:text=affects%20the%20RNGs.-,Cryptography,electronic%20commerce%2C%20etc.).](https://en.wikipedia.org/wiki/Applications_of_randomness#:~:text=affects%20the%20RNGs.-,Cryptography,electronic%20commerce%2C%20etc.\).)  
5. Understanding random number generators, and their limitations, in ..., accessed August 15, 2025, [https://www.redhat.com/en/blog/understanding-random-number-generators-and-their-limitations-linux](https://www.redhat.com/en/blog/understanding-random-number-generators-and-their-limitations-linux)  
6. Applications of randomness \- Wikipedia, accessed August 15, 2025, [https://en.wikipedia.org/wiki/Applications\_of\_randomness](https://en.wikipedia.org/wiki/Applications_of_randomness)  
7. 2\_P1\_Basics\_RNG\_handout.pdf  
8. MIT School of Engineering | » Can a computer generate a truly random number?, accessed August 15, 2025, [https://engineering.mit.edu/engage/ask-an-engineer/can-a-computer-generate-a-truly-random-number/](https://engineering.mit.edu/engage/ask-an-engineer/can-a-computer-generate-a-truly-random-number/)  
9. How tf do computers generate random numbers? : r/computerscience \- Reddit, accessed August 15, 2025, [https://www.reddit.com/r/computerscience/comments/1acir7n/how\_tf\_do\_computers\_generate\_random\_numbers/](https://www.reddit.com/r/computerscience/comments/1acir7n/how_tf_do_computers_generate_random_numbers/)  
10. What Is Monte Carlo Simulation? \- IBM, accessed August 15, 2025, [https://www.ibm.com/think/topics/monte-carlo-simulation](https://www.ibm.com/think/topics/monte-carlo-simulation)  
11. Monte Carlo Method Explained | Towards Data Science, accessed August 15, 2025, [https://towardsdatascience.com/monte-carlo-method-explained-8635edf2cf58/](https://towardsdatascience.com/monte-carlo-method-explained-8635edf2cf58/)  
12. What is The Monte Carlo Simulation? \- AWS, accessed August 15, 2025, [https://aws.amazon.com/what-is/monte-carlo-simulation/](https://aws.amazon.com/what-is/monte-carlo-simulation/)  
13. lumivero.com, accessed August 15, 2025, [https://lumivero.com/resources/monte-carlo-simulation-examples/\#:\~:text=Monte%20Carlo%20Simulation%20Examples%20in%20Finance\&text=For%20example%2C%20when%20debating%20whether,hedge%20against%20volatile%20exchange%20rates.](https://lumivero.com/resources/monte-carlo-simulation-examples/#:~:text=Monte%20Carlo%20Simulation%20Examples%20in%20Finance&text=For%20example%2C%20when%20debating%20whether,hedge%20against%20volatile%20exchange%20rates.)  
14. Monte Carlo Simulation \- YouTube, accessed August 15, 2025, [https://www.youtube.com/watch?v=7ESK5SaP-bc](https://www.youtube.com/watch?v=7ESK5SaP-bc)  
15. Monte Carlo Simulation: A Hands-On Guide \- Neptune.ai, accessed August 15, 2025, [https://neptune.ai/blog/monte-carlo-simulation](https://neptune.ai/blog/monte-carlo-simulation)  
16. Financial Modeling with Geometric Brownian Motion \- Scientific Research Publishing, accessed August 15, 2025, [https://www.scirp.org/journal/paperinformation?paperid=132102](https://www.scirp.org/journal/paperinformation?paperid=132102)  
17. The experiment that revealed the atomic world: Brownian Motion \- YouTube, accessed August 15, 2025, [https://www.youtube.com/watch?v=ZNzoTGv\_XiQ](https://www.youtube.com/watch?v=ZNzoTGv_XiQ)  
18. Brownian motion and diffusion processes | Actuarial Mathematics ..., accessed August 15, 2025, [https://library.fiveable.me/actuarial-mathematics/unit-2/brownian-motion-diffusion-processes/study-guide/rrXeyckVSpJj1bHe](https://library.fiveable.me/actuarial-mathematics/unit-2/brownian-motion-diffusion-processes/study-guide/rrXeyckVSpJj1bHe)  
19. Brownian Motion in Fluid Dynamics and Financial Markets: A Confluence of Physics and Finance \- Caio Marchesani \- Investment banker, accessed August 15, 2025, [https://caiomarchesani.com/articles/brownian-motion-in-fluid-dynamics-and-financial-markets-a-confluence-of-physics-and-finance/](https://caiomarchesani.com/articles/brownian-motion-in-fluid-dynamics-and-financial-markets-a-confluence-of-physics-and-finance/)  
20. Exploring the Role of Brownian Motion in Financial ... \- SciTePress, accessed August 15, 2025, [https://www.scitepress.org/Papers/2025/134463/134463.pdf](https://www.scitepress.org/Papers/2025/134463/134463.pdf)  
21. Cracking the Code of Decision Making: The Secret Behind the ..., accessed August 15, 2025, [https://medium.com/pythoneers/the-secretary-problem-how-to-optimize-your-chances-of-success-c18665184b8f](https://medium.com/pythoneers/the-secretary-problem-how-to-optimize-your-chances-of-success-c18665184b8f)  
22. Test Run \- The Secretary Problem | Microsoft Learn, accessed August 15, 2025, [https://learn.microsoft.com/en-us/archive/msdn-magazine/2016/september/test-run-the-secretary-problem](https://learn.microsoft.com/en-us/archive/msdn-magazine/2016/september/test-run-the-secretary-problem)  
23. Maximizing the chances of finding "the right one" by solving The Secretary Problem \- cdemi, accessed August 15, 2025, [https://blog.cdemi.io/maximizing-the-chances-of-finding-the-right-one-by-solving-the-secretary-problem/](https://blog.cdemi.io/maximizing-the-chances-of-finding-the-right-one-by-solving-the-secretary-problem/)  
24. Secretary Problem in Depth \- Number Analytics, accessed August 15, 2025, [https://www.numberanalytics.com/blog/secretary-problem-depth-approximation-algorithms](https://www.numberanalytics.com/blog/secretary-problem-depth-approximation-algorithms)  
25. The 37% Rule \- LifeNotes | Ali Abdaal, accessed August 15, 2025, [https://aliabdaal.com/newsletter/the-37-rule/](https://aliabdaal.com/newsletter/the-37-rule/)  
26. Probability \- YouTube, accessed August 15, 2025, [https://www.youtube.com/playlist?list=PLC58778F28211FA19](https://www.youtube.com/playlist?list=PLC58778F28211FA19)  
27. Probability Distribution: Definition & Calculations \- Statistics By Jim, accessed August 15, 2025, [https://statisticsbyjim.com/basics/probability-distributions/](https://statisticsbyjim.com/basics/probability-distributions/)