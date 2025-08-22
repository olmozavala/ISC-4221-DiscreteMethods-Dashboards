

# **A University-Level Tutorial on Data Mining: Clustering and Classification**

## **Introduction: What is Data Mining and Why Does It Matter?**

Welcome to this comprehensive tutorial on the foundational algorithms of data mining. As students of computer science, you are entering a world awash in data. The ability to extract meaningful patterns, structure, and knowledge from this data is no longer a niche skill but a fundamental component of modern computing. This module will equip you with the theoretical understanding and practical intuition behind two core pillars of data mining: clustering and classification.

### **Defining the Field**

The term "data mining" has a fascinating history. It was once used pejoratively by statisticians to describe the act of "dredging" through data to find spurious correlations, essentially torturing the data until it confessed to something, whether true or not.1 A famous example of this fallacy involved parapsychologist J.B. Rhine, who tested students for extrasensory perception (ESP) by having them guess the color of 10 cards. He found that about 1 in 1,000 students guessed all 10 correctly. Instead of recognizing this was statistically expected (

0.510≈1/1024), he declared them to have ESP. When he retested this group, they performed no better than average. His bizarre conclusion was that telling people they have ESP causes them to lose it.1

This historical context serves as a crucial warning. The tools of data mining are powerful, but without statistical rigor and domain expertise, they can easily lead to false conclusions. Today, however, the term has evolved to mean something far more principled. We define data mining as **the extraction of implicit, previously unknown, and potentially useful information from data**.1 This shift in meaning was driven by a fundamental shift in technology and society: the transition from a data-scarce to a data-abundant world.

### **Motivation \- The Data Deluge**

The core motivation for modern data mining is the exponential growth of information. It has been estimated that the amount of information in the world doubles every 20 months.1 This data comes from a staggering variety of sources:

* **Digital Life:** Every photo, movie, text message, email, and social media post contributes to this flood.1  
* **The Internet of Things (IoT):** Sensors in our infrastructure—water meters, oil pipelines, the power grid—generate continuous streams of data.1  
* **Scientific and Financial Data:** Satellite imagery, meteorological data, financial market transactions, medical and genomic testing, and the results of massive scientific simulations produce datasets of unimaginable scale.1

The critical realization is that very little of this raw data will ever be seen by human eyes. If it is to be understood at all, it must be analyzed by computers.1 Our task is to develop algorithms that can automatically find the patterns hidden within.

### **Real-World Impact**

Data mining is not an abstract academic exercise; it is the engine behind many of the technologies that shape our modern world.

* **Business Intelligence:** When Netflix recommends a movie or Spotify curates a playlist, they are using clustering and classification algorithms to group users with similar tastes.1 E-commerce companies use these techniques for  
  **customer segmentation**, identifying groups like "high-spending new customers" or "at-risk loyal customers" to tailor marketing campaigns.4 Airlines and hotels analyze traveler behavior to implement dynamic pricing strategies, maximizing revenue.1  
* **Science and Medicine:** In genomics, researchers have compared the genotypes of people with and without diabetes to discover a set of genes that collectively account for many cases of the disease.1 In healthcare, decision tree models are used to aid in medical diagnosis, helping doctors classify conditions based on patient symptoms and test results.6 Machine learning models interpret radiograms and other medical images, often spotting patterns invisible to the human eye.1  
* **Technology and Security:** Credit card companies and banks use data mining to perform **fraud detection**, identifying transaction patterns that deviate from a user's normal behavior.1 The software that powers autonomous vehicles and the natural language processing behind speech assistants are both sophisticated applications of data mining and machine learning.1

### **The Data Mining Process**

The algorithms discussed in this tutorial fit into a larger, multi-stage process. Understanding this context is key to appreciating their role 1:

1. **Data Gathering:** Collecting and consolidating raw data, perhaps from a data warehouse or by crawling the web.  
2. **Data Cleansing:** Identifying and correcting errors or bogus data (e.g., a patient's recorded fever of 125°C).  
3. **Feature Extraction:** Selecting and engineering the most relevant attributes from the raw data for the analysis.  
4. **Pattern Recognition and Discovery:** This is the core of data mining, where algorithms like clustering and classification are applied to find patterns.  
5. **Visualization:** Creating graphs and charts to help humans understand the discovered patterns.  
6. **Evaluation of Results:** Critically assessing the discovered patterns. Not every correlation is useful or even true. This final step, requiring human judgment, is the safeguard against the field's original sin of data dredging.

## **Part 1: Unsupervised Learning \- Finding Structure with Clustering**

We begin our exploration with **unsupervised learning**, a paradigm where we work with *unlabeled* data. The goal is not to predict a known outcome, but to discover the inherent structure within the data itself. The primary tool for this is **clustering**.

### **The Core Idea of Clustering: Grouping the Unknown**

Clustering is the task of identifying groups (or clusters) of objects that are more similar to each other than to objects in other groups.1 Imagine a hospital's patient records; clustering could automatically group patients with similar sets of symptoms, potentially revealing different subtypes of a disease.9 In marketing, it's used to segment customers into groups with similar purchasing habits, allowing for targeted advertising.4

This process is also known as **unsupervised learning** because we do not provide the algorithm with any pre-labeled examples. We simply give it the data and ask it to find the natural groupings.1

#### **The Distance Function**

To group similar objects, we first need a way to mathematically quantify "similarity." This is done using a **distance function**, often denoted as d(A,B), which measures how far apart two objects or clusters, A and B, are. A smaller distance implies greater similarity. Any valid distance function (or metric) must satisfy four key properties 1:

1. **Self:** The distance from an object to itself is zero. d(A,A)=0.  
2. **Commutative:** The distance from A to B is the same as the distance from B to A. d(A,B)=d(B,A).  
3. **Non-negative:** The distance between two distinct objects is always positive. d(A,B)≥0. (Note: In some contexts, d(A,B)=0 does not strictly imply A=B).  
4. **Triangle Inequality:** The shortest distance between two points is a straight line. d(A,C)≤d(A,B)+d(B,C).

For points in a standard geometric space, the most common metric is the Euclidean distance between two points p and q in Rn:

d(p,q)=(q1​−p1​)2+(q2​−p2​)2+⋯+(qn​−pn​)2​

However, many other distance functions exist for different types of data, such as binary data (e.g., comparing patient symptoms marked as 'yes' or 'no') or categorical data.9 The choice of distance function is a critical decision that depends on the nature of the data and the problem being solved.

### **Hierarchical Clustering: Building a Family Tree for Data**

One of the most intuitive ways to cluster data is to build a hierarchy, showing relationships at all scales, much like a biological family tree. This is the goal of **hierarchical clustering**. The result is a tree-like diagram called a **dendrogram**, which visualizes how data points are nested within clusters.1

There are two main approaches 1:

* **Agglomerative (Bottom-Up):** This is the more common method. It starts with each data point in its own cluster. Then, in each step, it merges the two "closest" clusters until only one cluster (containing all the data) remains.  
* **Divisive (Top-Down):** This method works in reverse. It starts with all data points in a single cluster and, at each step, splits a cluster into two until every data point is in its own cluster.

We will focus on the agglomerative approach. The central question for this algorithm is: how do we define the "closeness" between two clusters, especially when they contain multiple points? This is determined by the **linkage criterion**.

#### **Linkage Criteria: Defining "Closeness"**

There are four primary linkage criteria used to measure the distance between two clusters, A and B.1

* **Single Linkage (Nearest Neighbor):** The distance between two clusters is defined as the distance between the two *closest* points in the different clusters. Mathematically, if cluster A consists of points {αi​} and cluster B consists of points {βj​}, then d(A,B)=mind(αi​,βj​). This method is flexible and can identify clusters with non-elliptical shapes, but it is sensitive to outliers and can sometimes produce long, stringy clusters due to a phenomenon called the "chaining effect".1  
* **Complete Linkage (Farthest Neighbor):** This is the opposite of single linkage. The distance is defined by the two *farthest* points in the different clusters: d(A,B)=maxd(αi​,βj​). This method is less sensitive to outliers and tends to produce more compact, spherical clusters.1  
* **Average Linkage:** This method provides a compromise by defining the distance as the *average* of all pairwise distances between points in the two clusters: d(A,B)=∣A∣∣B∣1​∑i​∑j​d(αi​,βj​). It is less sensitive to outliers than single linkage but can still be biased towards finding spherical clusters.1  
* **Centroid Linkage:** This method calculates the geometric center (or centroid) of each cluster, αˉ and βˉ​. The distance between the clusters is then the Euclidean distance between their centroids: d(A,B)=∣∣αˉ−βˉ​∣∣2​.1

| Table 2: Comparison of Hierarchical Linkage Methods |  |  |  |
| :---- | :---- | :---- | :---- |
| **Linkage Method** | **Definition** | **Intuitive Description** | **Key Characteristics** |
| Single Linkage | d(A,B)=mind(αi​,βj​) | Distance between the two closest members. | Flexible, can find non-spherical shapes. Sensitive to outliers and noise (chaining effect). |
| Complete Linkage | d(A,B)=maxd(αi​,βj​) | Distance between the two farthest members. | Less sensitive to outliers. Tends to produce compact, spherical clusters. |
| Average Linkage | $d(A,B) \= \\frac{1}{ | A |  |
| Centroid Linkage | $d(A,B) \= |  | \\bar{\\alpha} \- \\bar{\\beta} |

#### **Worked Example & The Dendrogram**

Let's trace a simple agglomerative clustering example using single linkage.  
Suppose we have five points: A, B, C, D, E.

1. **Step 0: Compute the initial distance matrix.** We calculate the Euclidean distance between every pair of points. Let's say the smallest distance is between A and B.  
2. **Step 1: Merge A and B.** We form a new cluster (AB). We now have clusters (AB), C, D, E. We update the distance matrix. The distance from the new cluster (AB) to any other cluster, say C, is calculated using our linkage criterion (single linkage): d((AB),C)=min(d(A,C),d(B,C)).  
3. **Step 2: Merge again.** We find the new smallest distance in the updated matrix. Let's say it's between (AB) and C. We merge them to form cluster (ABC). We now have clusters (ABC), D, E.  
4. **Continue:** We repeat this process until all points are in a single cluster.

This merging process is visualized using a **dendrogram**.

*Diagram Description: A dendrogram looks like an upside-down tree. The individual data points (A, B, C, D, E) are at the bottom. As we move up, horizontal lines connect clusters that are being merged. The height of this horizontal line on the y-axis represents the distance at which the merge occurred. For example, the line connecting A and B would be very low, while the final line connecting all points would be at the top.*

A key feature of the dendrogram is that we can obtain a specific number of clusters by making a horizontal "cut" across it. If we cut the tree at a certain height, all the separate branches that the line intersects become our clusters.1 This allows us to explore different numbers of clusters without re-running the algorithm.

### **K-Means Clustering: Partitioning Data into K Groups**

While hierarchical clustering is excellent for exploring data structure, sometimes we have a more direct goal: partition the data into a specific number, *k*, of non-overlapping groups. This is the domain of **partitional clustering**, and the most famous algorithm for it is **K-Means**.1

#### **Motivation**

Imagine you are a data scientist at an e-commerce company. Management wants to launch three distinct marketing campaigns for three types of customers. Your task is to segment all customers into exactly three groups based on their purchasing behavior (e.g., frequency of purchase, average transaction value). This is a perfect application for K-Means, where *k=3*.4 Other applications include image compression, where all the colors in an image are clustered into

*k* representative colors to reduce file size.4

#### **The Mathematics Behind the Mean**

Before diving into the algorithm, let's consider a simple mathematical property of the average (or mean). For a set of one-dimensional points X={x1​,x2​,…,xn​}, the mean xˉ=n1​∑xi​ is the unique value c that minimizes the sum of squared Euclidean distances to all points in the set.1 This sum can be thought of as an "energy" or "cost" function:

E(c,X)=21​i=1∑n​(c−xi​)2

K-Means generalizes this idea. It seeks to find k cluster centers (centroids), {c1​,c2​,…,ck​}, that collectively minimize the total energy across all clusters. The total energy is the sum of squared distances from each point to its assigned cluster's centroid 1:  
E=21​j=1∑k​​xi​∈Cj​∑​∣∣cj​−xi​∣∣2​

where Cj​ is the set of points belonging to cluster j.  
The objective function itself provides deep clues about the algorithm's behavior. The use of squared Euclidean distance means that points far from a center contribute quadratically to the total error. This mathematical formulation inherently biases the algorithm toward finding compact, spherical clusters and makes it highly sensitive to outliers. A single outlier, being very far from any potential center, will have a massive squared distance, disproportionately pulling a centroid towards it during the optimization process.14

#### **Lloyd's Algorithm Step-by-Step**

Finding the absolute best set of centroids is computationally intractable. Instead, K-Means uses an iterative heuristic called **Lloyd's Algorithm** to find a good solution.1 The algorithm alternates between two main steps:

1. **Initialization:** Choose *k* initial points to serve as the first centroids. A common method is to simply pick *k* data points at random from the dataset. This random start is a critical point of weakness; a poor initialization can lead to a suboptimal clustering 1.  
2. **Assignment Step (Expectation):** For each data point in the dataset, calculate its distance to every centroid. Assign the data point to the cluster of its nearest centroid.  
3. **Update Step (Maximization):** After all points are assigned to clusters, recalculate the centroid for each cluster. The new centroid is the mean (average position) of all data points currently assigned to that cluster.  
4. **Repeat:** Iterate between the Assignment and Update steps. The algorithm has **converged** when, in an iteration, the cluster assignments no longer change (or the centroids move by a negligible amount).1

#### **Practical Considerations**

* **Convergence and Local Minima:** Lloyd's algorithm is guaranteed to converge, meaning it will always stop. Each step (both assignment and update) can be shown to decrease the total energy function E. However, it is not guaranteed to find the *global minimum*. It can easily get stuck in a **local minimum**, a solution that is good but not the best possible one.1 To mitigate this, the standard practice is to run the entire K-Means algorithm multiple times (e.g., 10 times), each with a different random initialization, and then select the clustering that resulted in the lowest total energy  
  E.1  
* **Choosing K:** The most common question in K-Means is: how do we choose the right value for *k*? If not specified by the problem, a popular heuristic is the **Elbow Method**. We run K-Means for a range of *k* values (e.g., *k* from 2 to 10\) and for each *k*, we plot the final total energy E. As *k* increases, the energy will always decrease. However, we look for an "elbow" in the plot—the point where the rate of decrease sharply flattens. This point suggests a value of *k* beyond which adding more clusters provides diminishing returns.1

#### **Weighted K-Means**

A simple but powerful extension is **Weighted K-Means**. In this version, each data point xi​ is given an importance weight wi​. This is useful in applications like deciding where to open a new chain of stores. The data points might be towns, but we want to weight them by their population. The algorithm is identical to standard K-Means, except the centroid update step becomes a weighted average, calculating the **center of mass** instead of the simple mean 1:

cj​=∑xi​∈Cj​​wi​∑xi​∈Cj​​wi​xi​​

### **Geometric Clustering: Voronoi Diagrams and Their Link to K-Means**

We now turn to a geometric perspective on clustering. A **Voronoi diagram** is a beautiful and fundamental geometric structure that partitions a plane into regions based on proximity to a given set of points, called **generators** or **sites**.1

#### **Motivation**

The classic motivating example comes from epidemiology. In 1854, London was struck by a cholera outbreak. Physician John Snow mapped the locations of deaths and water pumps. By observing that deaths clustered around the Broad Street pump, he identified it as the source of the contaminated water, a landmark event in public health.1 Snow's map was, in effect, a hand-drawn Voronoi diagram.

In the modern world, Voronoi diagrams have countless applications:

* **Networking:** Determining which cell tower your phone should connect to (the closest one).19  
* **Logistics:** Finding the nearest ambulance post to an emergency.1  
* **Computer Graphics & Gaming:** Procedurally generating organic-looking textures, terrain, or breakable geometry in video games.20  
* **Robotics:** Finding the clearest navigation path for an autonomous robot by staying as far away from known obstacles (the generators) as possible.22

#### **Formal Definition**

Given a set of *k* generator points {z1​,z2​,…,zk​} in a plane, the **Voronoi region** (or Voronoi cell) Vj​ associated with generator zj​ is the set of all points in the plane that are closer to zj​ than to any other generator zi​.1

Vj​={x∣d(x,zj​)≤d(x,zi​) for all i=j}

The collection of all k Voronoi regions is the Voronoi diagram. The boundaries of these regions are formed by the perpendicular bisectors of the lines connecting pairs of generators.1

#### **Delaunay Triangulation**

Closely related to the Voronoi diagram is its **dual graph**, the **Delaunay triangulation**. This is formed by drawing an edge between any two generators whose Voronoi regions share a common border. The result is a triangulation of the generator points that is widely used for mesh generation in computer graphics and scientific modeling.1

#### **The Bridge to K-Means: Centroidal Voronoi Tessellations (CVT)**

Here we arrive at a profound connection between the algebraic world of K-Means and the geometric world of Voronoi diagrams. A **Centroidal Voronoi Tessellation (CVT)** is a special, "balanced" Voronoi diagram where each generator point zi​ is also the geometric centroid (center of mass) of its own Voronoi region Vi​.1

This reveals a remarkable duality. The K-Means algorithm and the algorithm to compute a CVT are one and the same. Let's compare the steps:

| Lloyd's Algorithm for CVT | Lloyd's Algorithm for K-Means |
| :---- | :---- |
| 1\. Given a set of generators, construct their Voronoi diagram. | 1\. Given a set of centroids, assign each data point to its nearest centroid. |
| 2\. For each Voronoi region, calculate its geometric centroid. | 2\. For each cluster, calculate the mean of its member data points. |
| 3\. Move each generator to the centroid of its region. | 3\. Move each centroid to the mean of its cluster. |
| 4\. Repeat until generators no longer move. | 4\. Repeat until centroids no longer move. |

The first step in both algorithms is conceptually identical: partitioning the space (or the set of points) based on proximity to the current centers. The Voronoi region Vj​ *is* the set of all points closer to generator zj​. The second and third steps are also identical: finding the center of the newly formed regions and updating the generators/centroids.

Therefore, K-Means clustering of a set of discrete data points can be seen as the process of finding the optimal locations for *k* generators of a Centroidal Voronoi Tessellation for that data. The algebraic optimization problem of minimizing the K-Means energy function is equivalent to solving this geometric partitioning problem.1 This connection demonstrates a deep and elegant unity between seemingly disparate concepts in mathematics and computer science.

---

#### **Self-Check Questions**

1. What is the key difference between agglomerative and divisive hierarchical clustering?  
2. In K-Means, why is it standard practice to run the algorithm multiple times?  
3. What is a Centroidal Voronoi Tessellation (CVT), and how does its computation relate to the K-Means algorithm?

---

## **Part 2: Supervised Learning \- Making Predictions with Classifiers**

We now shift our focus from unsupervised to **supervised learning**. In this paradigm, we work with *labeled* data. Our goal is no longer to discover hidden structure, but to learn a function that can predict the label for new, unseen data. The algorithms that perform this task are called **classifiers**.

### **The Shift to Classification: From Discovery to Prediction**

The fundamental difference between clustering and classification lies in the data and the goal. Clustering takes unlabeled data and discovers groups. Classification takes labeled data (the **training set**) and learns a model to assign labels to new, unlabeled data (the **test set**).1

| Table 1: Clustering vs. Classification |  |  |
| :---- | :---- | :---- |
| **Criterion** | **Clustering (Unsupervised)** | **Classification (Supervised)** |
| **Prior Knowledge of Classes** | No | Yes |
| **Goal / Use Case** | Discover and suggest groups based on patterns in data. | Classify new samples into known, predefined classes. |
| **Data Needs** | Unlabeled samples. | Labeled samples (training data). |
| **Example Algorithms** | K-Means, Hierarchical Clustering | Decision Trees, Neural Networks |

#### **Terminology**

To discuss classification, we need a precise vocabulary 1:

* **Instance:** A single data record (e.g., one patient, one email).  
* **Attribute Set (Features):** The set of measurements or characteristics describing an instance, denoted by x.  
* **Class Label:** The predefined category to which an instance belongs, denoted by y.  
* **Training Set:** A collection of instances where both the attribute set x and the class label y are known.  
* **Classification Model:** A function f(x) learned from the training set that maps an attribute set to a predicted class label. This is also called the target function.  
* **Test Set:** A collection of instances, unseen during training, used to evaluate the performance of the model.

#### **Attribute Types**

Attributes can be of various types, which can influence the choice of classification algorithm 1:

* **Nominal:** Categorical values with no inherent order (e.g., eye\_color \= {'blue', 'brown', 'green'}).  
* **Ordinal:** Categorical values with a meaningful order (e.g., temperature \= {'cold', 'mild', 'hot'}).  
* **Interval:** Numerical values where differences are meaningful, but there is no true zero (e.g., temperature in Celsius, dates).  
* **Ratio:** Numerical values where both differences and ratios are meaningful, with a true zero point (e.g., age, income).

Many classic classification algorithms, including the decision trees we will study next, are best suited for data with nominal or binary attributes.

### **Decision Trees: Making Decisions with a Flowchart**

One of the most intuitive and powerful classification models is the **decision tree**. It creates a model that predicts the value of a target variable by learning simple decision rules inferred from the data features, presented in a flowchart-like structure.1

#### **Motivation**

Decision trees are highly valued because they are **interpretable**. Unlike some complex models that act as "black boxes," the logic of a decision tree is transparent and easy for humans to understand.24 This is critical in high-stakes domains:

* **Medical Diagnosis:** A doctor might use a decision tree to diagnose a condition. The tree could ask a series of questions like "Does the patient have a fever?" and "Is the cough dry or productive?" to arrive at a probable diagnosis. The doctor can trace the logic, making it a trustworthy aid.6  
* **Financial Analysis:** A bank can use a decision tree to decide whether to approve a loan. The model might check credit\_score \> 700, then income \> $50,000, etc. The reasons for an approval or denial are explicit.1

#### **Structure of a Decision Tree**

A decision tree has three main types of nodes 1:

* **Root Node:** The top-most node, representing the entire training dataset. It has no incoming branches.  
* **Internal (Decision) Node:** Represents a test on an attribute (e.g., "Is outlook sunny?"). It has exactly one incoming branch and two or more outgoing branches, one for each possible outcome of the test.  
* **Leaf (Terminal) Node:** Represents a class label (a final decision, e.g., "Play Tennis"). It has one incoming branch and no outgoing branches.

To classify a new instance, one starts at the root and traverses the tree by following the branches corresponding to the instance's attribute values until a leaf node is reached. The label of that leaf node is the predicted class.

#### **Building a Tree: Hunt's Algorithm**

Finding the globally optimal decision tree is an NP-complete problem. Therefore, in practice, trees are built using a **greedy**, top-down, recursive algorithm, a strategy often based on **Hunt's Algorithm**.1

Let Dt​ be the set of training records associated with a node t. The basic algorithm is as follows 1:

1. If all records in Dt​ belong to the same class y, then t is a leaf node labeled as y.  
2. If Dt​ contains records belonging to more than one class, use an **attribute selection measure** to find the "best" way to split the records.  
3. Label node t with the chosen attribute. For each outcome of the split, create a new child node and distribute the records in Dt​ to the appropriate child.  
4. Apply the algorithm recursively to each child node.

The most important question in this process is in Step 2: how do we mathematically define the "best" split?

### **The Mathematics of a "Good" Split: Impurity and Information Gain**

The greedy strategy of a decision tree is to choose the split that results in the "purest" child nodes—that is, nodes that are as homogeneous as possible with respect to the class labels.1 To do this, we need a way to measure the

**impurity** of a set of data.

The concept of entropy, borrowed from information theory, provides a robust mathematical foundation for this task. Information theory defines entropy as the average level of uncertainty or surprise in a variable's outcomes. A dataset with a 50/50 mix of "Yes" and "No" labels has maximum uncertainty (high entropy), while a dataset with only "Yes" labels has zero uncertainty (zero entropy). The goal of a split is to ask an informative question that reduces this uncertainty.

#### **Impurity Measures**

Let p(i∣t) be the fraction of records belonging to class i at a given node t. There are three common measures of impurity for a node 1:

1. Entropy: Measures the level of uncertainty in a node. A pure node has an entropy of 0\.

   Entropy(t)=−i=1∑k​p(i∣t)log2​p(i∣t)  
2. Gini Impurity: Measures the probability of misclassifying a randomly chosen element if it were randomly labeled according to the class distribution at the node. A pure node has a Gini index of 0\.

   Gini(t)=1−i=1∑k​p(i∣t)2  
3. Classification Error: The simplest measure, representing the error rate if we assigned every record to the majority class at the node.

   Error(t)=1−imax​p(i∣t)

For a two-class problem, all three measures are 0 when the node is pure (p(1)=1 or p(1)=0) and reach their maximum when the classes are perfectly balanced (p(1)=0.5).1

#### **Information Gain**

To evaluate a split, we measure the change in impurity. **Information Gain** is the reduction in impurity achieved by partitioning a dataset based on an attribute. The attribute that provides the largest information gain is chosen for the split.1

The gain, Δ, is calculated as the impurity of the parent node minus the weighted average of the impurities of the child nodes:

Δ=I(parent)−j=1∑k​NNj​​I(childj​)

where I is an impurity measure (like Entropy), N is the total number of records at the parent, and Nj​ is the number of records in the j-th child node.1 When Entropy is used, this is specifically called  
**information gain**.

#### **Worked Example: The "Play Tennis" Dataset**

Let's use the classic "Play Tennis" dataset from the slides to see this in action. The parent node contains 14 records: 9 "Play" and 5 "Don't Play".

1. Calculate Parent Entropy:  
   p(Play)=9/14, p(Don’t Play)=5/14.  
   Entropy(parent)=−(149​log2​149​+145​log2​145​)=0.940  
2. **Evaluate Split on 'Outlook':** The 'Outlook' attribute has three values: Sunny, Overcast, Rain.  
   * **Sunny:** 5 records (2 Play, 3 Don't Play). Entropy(Sunny)=0.971.  
   * **Overcast:** 4 records (4 Play, 0 Don't Play). Entropy(Overcast)=0.  
   * **Rain:** 5 records (3 Play, 2 Don't Play). Entropy(Rain)=0.971.  
3. Calculate Information Gain for 'Outlook':  
   Weighted average entropy of children \= 145​Entropy(Sunny)+144​Entropy(Overcast)+145​Entropy(Rain)  
   \=145​(0.971)+144​(0)+145​(0.971)=0.693  
   Gain(Outlook)=Entropy(parent)−0.693=0.940−0.693=0.247  
4. **Repeat for Other Attributes:** We would perform the same calculation for 'Temperature', 'Humidity', and 'Windy'. The calculations show that these attributes result in gains of 0.029, 0.152, and 0.048, respectively.1  
5. **Conclusion:** Since 'Outlook' provides the highest information gain (0.247), it is chosen as the splitting attribute for the root node. The algorithm would then recurse on the 'Sunny' and 'Rain' child nodes, as the 'Overcast' node is already pure.

### **A Glimpse into Neural Networks: The "Black Box" Classifier**

While decision trees are powerful and interpretable, another class of models, **Artificial Neural Networks (ANNs)**, often achieves state-of-the-art performance, especially on complex, unstructured data like images, audio, and text.29

#### **Motivation and Analogy**

ANNs are computational models loosely inspired by the structure of the human brain, consisting of interconnected nodes called "neurons" organized in layers.29 They excel at learning intricate, non-linear patterns from vast amounts of data.

To understand the layered structure, consider an analogy of a school system teaching a complex subject 32:

* **Input Layer (Elementary School):** This layer receives the raw data (the features of an instance), like elementary school teachers who introduce students to the basic vocabulary of a subject.  
* **Hidden Layers (Middle & High School):** These are the intermediate layers. Each layer takes the output from the previous one and learns progressively more complex and abstract features. A first hidden layer might learn to recognize simple patterns (like edges in an image), while deeper layers combine these to recognize more complex concepts (like shapes, then objects).  
* **Output Layer (Final Exam):** This final layer takes the highly processed information from the last hidden layer and produces the final prediction (the class label).

#### **The Artificial Neuron**

The fundamental processing unit of a network is the artificial neuron. It performs a simple two-step computation 33:

1. **Calculate a Weighted Sum:** The neuron receives multiple inputs (x1​,x2​,…,xn​). Each input is multiplied by a **weight** (w1​,w2​,…,wn​), which signifies its importance. These weighted inputs are summed together, along with a **bias** term b. The result is z=(∑wi​xi​)+b.  
2. **Apply an Activation Function:** The sum z is then passed through a non-linear **activation function**, such as a Sigmoid or ReLU function. This function transforms the sum into the neuron's final output. The non-linearity is crucial; it is what allows the network to learn patterns that are far more complex than simple linear relationships.

#### **The Learning Process**

A neural network "learns" by finding the optimal set of weights and biases that allow it to correctly map inputs to outputs. This is typically done through a process called **backpropagation** 29:

1. **Feedforward:** An instance from the training set is fed into the input layer. The activations flow forward through the network, layer by layer, until the output layer produces a prediction.  
2. **Loss Calculation:** The network's prediction is compared to the true label from the training data using a **loss function**, which quantifies the error or "loss."  
3. **Backpropagation:** The error is propagated backward through the network. Using calculus (specifically, the chain rule), the algorithm calculates how much each weight and bias contributed to the total error.  
4. **Weight Update:** The weights and biases are adjusted slightly in the direction that will reduce the error.

This process is repeated for thousands or millions of training instances, with the network gradually adjusting its weights to become a better and better predictor.

#### **Decision Trees vs. Neural Networks**

The choice between a decision tree and a neural network involves a fundamental trade-off between interpretability and predictive power. A decision tree's logic is explicit; one can follow the path of any prediction and understand the rules that led to it. This makes it a "white box" model, which is essential in fields where accountability is paramount.24

A neural network, by contrast, is often called a "black box".36 Its predictive knowledge is encoded and distributed across millions of numerical weights in a complex, non-linear system. While it may make highly accurate predictions, it is often impossible to articulate

*why* it made a specific decision in a way that is intuitive to humans. This has given rise to a major field of research in modern AI called **Explainable AI (XAI)**, which seeks to develop techniques to peer inside these black boxes.

| Table 3: Decision Trees vs. Neural Networks |  |  |
| :---- | :---- | :---- |
| **Aspect** | **Decision Trees** | **Neural Networks** |
| **Interpretability** | High ("White Box"). The decision path is a clear set of rules. | Low ("Black Box"). Decisions emerge from complex interactions of weights. |
| **Typical Data Type** | Excels at structured, tabular data. Handles mixed numerical/categorical data well. | Excels at unstructured data (images, audio, text). |
| **Data Volume** | Can work well with smaller datasets. | Requires large amounts of data to train effectively. |
| **Feature Preparation** | Less sensitive to feature scaling. | Highly sensitive; requires data normalization/standardization. |
| **Computational Cost** | Generally faster to train. | Can be computationally very expensive and time-consuming to train. |

---

#### **Self-Check Questions**

1. What is the core difference between supervised and unsupervised learning?  
2. In a decision tree, what is the purpose of an impurity measure like Entropy or Gini Impurity?  
3. Why are neural networks often referred to as "black box" models, and why is this a concern in some applications?

---

## **Key Takeaways**

* **Data Mining** is the principled, automated process of extracting useful, previously unknown information from large datasets.  
* **Unsupervised Learning (Clustering)** finds inherent groupings in unlabeled data.  
  * **Hierarchical Clustering** builds a tree-like dendrogram to show nested relationships, using linkage criteria (single, complete, etc.) to define inter-cluster distance.  
  * **K-Means Clustering** partitions data into a pre-specified *k* number of clusters by iteratively assigning points to the nearest centroid and updating the centroid's position.  
  * **Voronoi Diagrams** partition a plane based on proximity to generator points. The K-Means algorithm is equivalent to the method for finding a **Centroidal Voronoi Tessellation (CVT)**, linking algebraic optimization with geometric partitioning.  
* **Supervised Learning (Classification)** learns from labeled data to predict the class of new, unseen data.  
  * **Decision Trees** are interpretable, flowchart-like models that make predictions by recursively splitting data based on the attribute that provides the highest **Information Gain** (greatest reduction in impurity).  
  * **Impurity Measures** (Entropy, Gini, Classification Error) are mathematical functions used to quantify the homogeneity of a set of labeled data, guiding the tree's construction.  
  * **Neural Networks** are powerful but complex "black box" models, inspired by the brain, that excel at learning intricate patterns in large, unstructured datasets. They learn by adjusting internal weights via backpropagation.  
* There is often a **trade-off between model interpretability and predictive power**. Simple models like decision trees are easy to understand, while complex models like neural networks may be more accurate but are difficult to explain.

## **Practice Problems**

Here are some problems to test your understanding of the concepts covered in this tutorial.

**Problem 1: Conceptual Questions**

(a) Explain why the random initialization of centroids in the K-Means algorithm can lead to different final clustering results. Why is running the algorithm multiple times a common solution?

(b) A data scientist is using single-linkage hierarchical clustering on a dataset that contains two distinct, dense clusters connected by a thin "bridge" of outlier points. Describe the likely result of the clustering. How would the result differ if they used complete-linkage instead?

(c) A bank wants to build a model to approve or deny loan applications. They need to be able to explain to regulators exactly why any given application was denied. Between a decision tree and a neural network, which model would be more appropriate and why?

**Problem 2: K-Means Algorithm Tracing**

Consider the following 2D data points: A=(1,1), B=(2,1), C=(1,2), D=(5,4), E=(6,4), F=(5,5).  
You want to cluster these points into k=2 clusters. The initial centroids are c1​=A=(1,1) and c2​=D=(5,4).  
Trace the first two full iterations of the K-Means (Lloyd's) algorithm. For each iteration, show:

1. The assignment of each point (A, B, C, D, E, F) to either cluster 1 or cluster 2 based on Euclidean distance.  
2. The calculation of the new centroids based on the updated cluster memberships.

**Problem 3: Decision Tree \- Root Node Selection**

You are given the following dataset to predict whether a student will pass an exam based on whether they studied and whether they slept well.

| Studied | Slept Well | Pass |
| :---- | :---- | :---- |
| Yes | Yes | Yes |
| Yes | No | Yes |
| Yes | Yes | Yes |
| No | Yes | No |
| No | No | No |
| No | No | No |
| Yes | No | No |
| No | Yes | Yes |

Using the **Gini Impurity** as your impurity measure, calculate the information gain for splitting on 'Studied' versus splitting on 'Slept Well'. Which attribute should be chosen as the root node of the decision tree?

**Problem 4: Short Proof**

Prove that for a set of one-dimensional points X={x1​,x2​,…,xn​}, the sample mean xˉ=n1​∑i=1n​xi​ is the value c that minimizes the energy function E(c,X)=21​∑i=1n​(c−xi​)2. (Hint: Use calculus. Find the derivative of E with respect to c and set it to zero).

**Problem 5: Connecting Concepts**

Explain the relationship between the "Assignment Step" of the K-Means algorithm and the definition of a Voronoi diagram.

---

### **Solutions and Hints**

**Solution 1: Conceptual Questions**

(a) K-Means converges to a local minimum of the energy function, not necessarily the global minimum. The final local minimum it finds depends on the starting positions of the centroids. Different random initializations can place the centroids in different "basins of attraction," leading to different, and potentially suboptimal, final clusters. By running the algorithm many times with different random starts and choosing the result with the lowest overall energy (sum of squared distances), we increase the probability of finding a solution that is at or close to the global minimum.

(b) With **single-linkage**, the algorithm will likely group the two dense clusters into one large cluster early on. Because single-linkage defines cluster distance by the *closest* points, the "bridge" of outliers will act as a chain, connecting the two otherwise distinct groups. The result will be one large, elongated cluster. With **complete-linkage**, which defines distance by the *farthest* points, the bridge points will not be enough to merge the main clusters. The algorithm will keep the two dense clusters separate for much longer, as the distance between their farthest points is large. The result will be two compact clusters and some outliers that are merged in later.

(c) The **decision tree** is far more appropriate. Its primary advantage is interpretability. The path from the root to any leaf node provides a clear, explicit set of if-then rules that led to the final decision (e.g., "IF credit\_score \< 600 AND debt\_to\_income\_ratio \> 0.5 THEN Deny"). This provides the exact, auditable reason required by regulators. A neural network is a "black box"; while it might be accurate, explaining its decision would be extremely difficult.

**Solution 2: K-Means Algorithm Tracing**

**Iteration 1:**

* **Initial Centroids:** c1​=(1,1), c2​=(5,4).  
* **Assignment Step:**  
  * Dist(A, c1​)=0, Dist(A, c2​)=42+32​=5. Assign A to Cluster 1\.  
  * Dist(B, c1​)=1, Dist(B, c2​)=32+32​≈4.24. Assign B to Cluster 1\.  
  * Dist(C, c1​)=1, Dist(C, c2​)=42+22​≈4.47. Assign C to Cluster 1\.  
  * Dist(D, c1​)=5, Dist(D, c2​)=0. Assign D to Cluster 2\.  
  * Dist(E, c1​)=52+32​≈5.83, Dist(E, c2​)=1. Assign E to Cluster 2\.  
  * Dist(F, c1​)=42+42​≈5.66, Dist(F, c2​)=1. Assign F to Cluster 2\.  
* **Cluster Memberships:** Cluster 1 \= {A, B, C}, Cluster 2 \= {D, E, F}.  
* **Update Step (New Centroids):**  
  * c1′​=(31+2+1​,31+1+2​)=(4/3,4/3)≈(1.33,1.33).  
  * c2′​=(35+6+5​,34+4+5​)=(16/3,13/3)≈(5.33,4.33).

**Iteration 2:**

* **Current Centroids:** c1​=(1.33,1.33), c2​=(5.33,4.33).  
* **Assignment Step:**  
  * A=(1,1) is closer to c1​. Assign A to Cluster 1\.  
  * B=(2,1) is closer to c1​. Assign B to Cluster 1\.  
  * C=(1,2) is closer to c1​. Assign C to Cluster 1\.  
  * D=(5,4) is closer to c2​. Assign D to Cluster 2\.  
  * E=(6,4) is closer to c2​. Assign E to Cluster 2\.  
  * F=(5,5) is closer to c2​. Assign F to Cluster 2\.  
* **Cluster Memberships:** Cluster 1 \= {A, B, C}, Cluster 2 \= {D, E, F}. The assignments did not change.  
* **Update Step:** The new centroids will be the same as the previous iteration.  
* **Conclusion:** The algorithm has converged after the first full iteration.

**Solution 3: Decision Tree \- Root Node Selection**

Total records \= 8\. Pass=4, No Pass=4.  
Gini(parent)=1−((4/8)2+(4/8)2)=1−(0.25+0.25)=0.5.  
**Split on 'Studied':**

* Studied=Yes (4 records): 3 Pass, 1 No Pass.  
  Gini(Studied=Yes)=1−((3/4)2+(1/4)2)=1−(0.5625+0.0625)=0.375.  
* Studied=No (4 records): 1 Pass, 3 No Pass.  
  Gini(Studied=No)=1−((1/4)2+(3/4)2)=1−(0.0625+0.5625)=0.375.  
* **Weighted Gini:** 84​(0.375)+84​(0.375)=0.375.  
* **Gain('Studied'):** 0.5−0.375=0.125.

**Split on 'Slept Well':**

* Slept Well=Yes (4 records): 3 Pass, 1 No Pass.  
  Gini(Slept Well=Yes)=1−((3/4)2+(1/4)2)=0.375.  
* Slept Well=No (4 records): 1 Pass, 3 No Pass.  
  Gini(Slept Well=No)=1−((1/4)2+(3/4)2)=0.375.  
* **Weighted Gini:** 84​(0.375)+84​(0.375)=0.375.  
* **Gain('Slept Well'):** 0.5−0.375=0.125.

**Conclusion:** Both attributes provide the same information gain (0.125). The algorithm could choose either one to be the root node.

**Solution 4: Short Proof**

To find the value of c that minimizes E(c,X), we take the derivative of E with respect to c and set it to 0\.

E(c,X)=21​i=1∑n​(c−xi​)2  
$$\\frac{\\partial\\mathcal{E}}{\\partial c} \= \\frac{1}{2}\\sum\_{i=1}^{n} 2(c-x\_i) \\cdot 1 \= \\sum\_{i=1}^{n}(c-x\_i)$$Set the derivative to zero to find the critical point:$$\\sum\_{i=1}^{n}(c-x\_i) \= 0$$$$\\sum\_{i=1}^{n}c \- \\sum\_{i=1}^{n}x\_i \= 0$$$$nc \- \\sum\_{i=1}^{n}x\_i \= 0$$$$nc \= \\sum\_{i=1}^{n}x\_i$$$$c \= \\frac{1}{n}\\sum\_{i=1}^{n}x\_i \= \\bar{x}$$

The second derivative is ∂c2∂2E​=∑i=1n​1=n, which is positive, confirming this critical point is a minimum. Thus, the sample mean xˉ minimizes the energy function.  
**Solution 5: Connecting Concepts**

The "Assignment Step" of K-Means requires assigning each data point to the cluster whose centroid is closest. A Voronoi diagram is a partition of a plane into regions, where each region Vj​ contains all points that are closer to its generator point zj​ than to any other generator. Therefore, the K-Means assignment step is computationally equivalent to constructing a Voronoi diagram where the centroids act as the generator points. The set of all data points assigned to Cluster 1 is precisely the set of data points that fall within the Voronoi region of centroid c1​.

## **Extension Questions**

1. **DBSCAN vs. K-Means:** Research the clustering algorithm DBSCAN (Density-Based Spatial Clustering of Applications with Noise). What are its main parameters (eps and min\_pts)? Describe two specific types of datasets where DBSCAN would significantly outperform K-Means.  
2. **Ensemble Methods:** A single decision tree is prone to overfitting. A common technique to improve performance is to use an "ensemble method" like a Random Forest. At a high level, how does a Random Forest work, and why does this technique typically lead to a more robust and accurate model than a single, deep decision tree?

## **Extra Learning**

### **High-Quality YouTube Videos**

1. **StatQuest with Josh Starmer: K-means clustering**  
   * **Link:** [https://www.youtube.com/watch?v=4b5d3muPQmA](https://www.youtube.com/watch?v=4b5d3muPQmA)  
   * **Note:** An excellent, intuitive visual explanation of the K-Means algorithm, perfect for building a foundational understanding. Josh Starmer has a gift for making complex topics clear and accessible.  
2. **Serrano.Academy: Clustering: K-means and Hierarchical**  
   * **Link:** [https://www.youtube.com/watch?v=QXOkPvFM6NU](https://www.youtube.com/watch?v=QXOkPvFM6NU)  
   * **Note:** This video by Luis Serrano uses wonderful animations and analogies to explain both K-Means and Hierarchical clustering, including dendrograms and the elbow method. It's great for visual learners.

### **Reputable Blog Posts and Online Articles**

1. **GeeksForGeeks: Decision Tree Introduction with Example**  
   * **Link:** [https://www.geeksforgeeks.org/machine-learning/decision-tree-introduction-example/](https://www.geeksforgeeks.org/machine-learning/decision-tree-introduction-example/)  
   * **Note:** A clear, step-by-step written tutorial that walks through the concepts of Information Gain and Gini Index with a worked example. It's a great text-based resource to reinforce the material covered in our tutorial.  
2. **Towards Data Science: Voronoi Grids: A Practical Application**  
   * **Link:** [https://towardsdatascience.com/voronoi-grids-a-practical-application-7e6ee3b1daf0](https://towardsdatascience.com/voronoi-grids-a-practical-application-7e6ee3b1daf0)  
   * **Note:** This article shows a real-world application of Voronoi diagrams using Python to map school zones in Melbourne. It's a great example of how these geometric concepts are applied to solve practical geospatial problems and includes code examples.

#### **Works cited**

1. 5\_P1\_Data\_Mining\_handout.pdf  
2. How Netflix Uses Customer Segmentation To Deliver Personalized Recommendations And Content \- FasterCapital, accessed August 16, 2025, [https://fastercapital.com/topics/how-netflix-uses-customer-segmentation-to-deliver-personalized-recommendations-and-content.html/1](https://fastercapital.com/topics/how-netflix-uses-customer-segmentation-to-deliver-personalized-recommendations-and-content.html/1)  
3. Case Study: How Spotify Prioritizes Data Projects for a Personalized Music Experience | Pragmatic Institute, accessed August 16, 2025, [https://www.pragmaticinstitute.com/resources/articles/data/case-study-how-spotify-prioritizes-data-projects-for-a-personalized-music-experience/](https://www.pragmaticinstitute.com/resources/articles/data/case-study-how-spotify-prioritizes-data-projects-for-a-personalized-music-experience/)  
4. Real-World Applications of the K-Means Algorithm in Data Science \- SkillCamper, accessed August 16, 2025, [https://www.skillcamper.com/blog/real-world-applications-of-the-k-means-algorithm-in-data-science](https://www.skillcamper.com/blog/real-world-applications-of-the-k-means-algorithm-in-data-science)  
5. Customer Segmentation Using the K-Means Clustering Algorithm \- Federal Polytechnic Ilaro, accessed August 16, 2025, [https://sciencetechjournal.federalpolyilaro.edu.ng/storage/article/JST%2015%20Done\_1729671956.pdf](https://sciencetechjournal.federalpolyilaro.edu.ng/storage/article/JST%2015%20Done_1729671956.pdf)  
6. opencv-machine-learning/notebooks/05.00-Using-Decision-Trees-to-Make-a-Medical-Diagnosis.ipynb at master \- GitHub, accessed August 16, 2025, [https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/05.00-Using-Decision-Trees-to-Make-a-Medical-Diagnosis.ipynb](https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/05.00-Using-Decision-Trees-to-Make-a-Medical-Diagnosis.ipynb)  
7. A Review on Decision Tree Algorithm in Healthcare Applications, accessed August 16, 2025, [http://ijcs.net/ijcs/index.php/ijcs/article/view/4026](http://ijcs.net/ijcs/index.php/ijcs/article/view/4026)  
8. Why: Practical Applications of Decision Trees (Part 2\) | by Ankush Singh | Medium, accessed August 16, 2025, [https://medium.com/@diehardankush/why-practical-applications-of-decision-trees-ae09e04b2b16](https://medium.com/@diehardankush/why-practical-applications-of-decision-trees-ae09e04b2b16)  
9. Hierarchical Cluster Analysis | Blogs \- Sigma Magic, accessed August 16, 2025, [https://www.sigmamagic.com/blogs/hierarchical-clustering/](https://www.sigmamagic.com/blogs/hierarchical-clustering/)  
10. What is Hierarchical Clustering? \- IBM, accessed August 16, 2025, [https://www.ibm.com/think/topics/hierarchical-clustering](https://www.ibm.com/think/topics/hierarchical-clustering)  
11. Hierarchical Clustering in Machine Learning \- GeeksforGeeks, accessed August 16, 2025, [https://www.geeksforgeeks.org/machine-learning/hierarchical-clustering/](https://www.geeksforgeeks.org/machine-learning/hierarchical-clustering/)  
12. Hierarchical Clustering \- LearnDataSci, accessed August 16, 2025, [https://www.learndatasci.com/glossary/hierarchical-clustering/](https://www.learndatasci.com/glossary/hierarchical-clustering/)  
13. K means Clustering – Introduction \- GeeksforGeeks, accessed August 16, 2025, [https://www.geeksforgeeks.org/machine-learning/k-means-clustering-introduction/](https://www.geeksforgeeks.org/machine-learning/k-means-clustering-introduction/)  
14. 40 Questions to test a Data Scientist on Clustering Techniques Flashcards | Quizlet, accessed August 16, 2025, [https://quizlet.com/300367120/40-questions-to-test-a-data-scientist-on-clustering-techniques-flash-cards/](https://quizlet.com/300367120/40-questions-to-test-a-data-scientist-on-clustering-techniques-flash-cards/)  
15. Top 7 K-Means Clustering Techniques to Enhance Analysis \- Number Analytics, accessed August 16, 2025, [https://www.numberanalytics.com/blog/top-7-k-means-clustering-techniques-to-enhance-analysis](https://www.numberanalytics.com/blog/top-7-k-means-clustering-techniques-to-enhance-analysis)  
16. Simple Explanation of the K-Means Unsupervised Learning Algorithm \- YouTube, accessed August 16, 2025, [https://www.youtube.com/watch?v=PJGSEttUzx8](https://www.youtube.com/watch?v=PJGSEttUzx8)  
17. www.analyticsvidhya.com, accessed August 16, 2025, [https://www.analyticsvidhya.com/blog/2024/01/a-quick-overview-of-voronoi-diagrams/\#:\~:text=Applications%20of%20Voronoi%20Diagrams%20in%20Various%20Fields\&text=In%20computer%20graphics%20and%20visualization%2C%20they%20generate%20realistic%20textures%2C%20simulate,map%20overlay%2C%20and%20network%20planning.](https://www.analyticsvidhya.com/blog/2024/01/a-quick-overview-of-voronoi-diagrams/#:~:text=Applications%20of%20Voronoi%20Diagrams%20in%20Various%20Fields&text=In%20computer%20graphics%20and%20visualization%2C%20they%20generate%20realistic%20textures%2C%20simulate,map%20overlay%2C%20and%20network%20planning.)  
18. Voronoi Grids: A Practical Application | Towards Data Science, accessed August 16, 2025, [https://towardsdatascience.com/voronoi-grids-a-practical-application-7e6ee3b1daf0/](https://towardsdatascience.com/voronoi-grids-a-practical-application-7e6ee3b1daf0/)  
19. How to create Voronoi regions with Geospatial data in Python, accessed August 16, 2025, [https://towardsdatascience.com/how-to-create-voronoi-regions-with-geospatial-data-in-python-adbb6c5f2134/](https://towardsdatascience.com/how-to-create-voronoi-regions-with-geospatial-data-in-python-adbb6c5f2134/)  
20. Voronoi Diagram \- Procedural Content Generation Wiki, accessed August 16, 2025, [http://pcg.wikidot.com/pcg-algorithm:voronoi-diagram](http://pcg.wikidot.com/pcg-algorithm:voronoi-diagram)  
21. Heightmaps and Voronoi Diagrams: Revolutionizing Game World Generation \- Wayline, accessed August 16, 2025, [https://www.wayline.io/blog/heightmaps-voronoi-diagrams-game-world-generation](https://www.wayline.io/blog/heightmaps-voronoi-diagrams-game-world-generation)  
22. Voronoi diagram \- Wikipedia, accessed August 16, 2025, [https://en.wikipedia.org/wiki/Voronoi\_diagram](https://en.wikipedia.org/wiki/Voronoi_diagram)  
23. A Quick Overview of Voronoi Diagrams \- Analytics Vidhya, accessed August 16, 2025, [https://www.analyticsvidhya.com/blog/2024/01/a-quick-overview-of-voronoi-diagrams/](https://www.analyticsvidhya.com/blog/2024/01/a-quick-overview-of-voronoi-diagrams/)  
24. What is a Decision Tree? \- IBM, accessed August 16, 2025, [https://www.ibm.com/think/topics/decision-trees](https://www.ibm.com/think/topics/decision-trees)  
25. Decision tree learning \- Wikipedia, accessed August 16, 2025, [https://en.wikipedia.org/wiki/Decision\_tree\_learning](https://en.wikipedia.org/wiki/Decision_tree_learning)  
26. Decision tree An example of decision tree, which can classify each... \- ResearchGate, accessed August 16, 2025, [https://www.researchgate.net/figure/Decision-tree-An-example-of-decision-tree-which-can-classify-each-patient-as-healthy\_fig3\_330298818](https://www.researchgate.net/figure/Decision-tree-An-example-of-decision-tree-which-can-classify-each-patient-as-healthy_fig3_330298818)  
27. Demystifying Decision Trees for the Real World \- KDnuggets, accessed August 16, 2025, [https://www.kdnuggets.com/demystifying-decision-trees-for-the-real-world](https://www.kdnuggets.com/demystifying-decision-trees-for-the-real-world)  
28. Decision Tree \- GeeksforGeeks, accessed August 16, 2025, [https://www.geeksforgeeks.org/machine-learning/decision-tree/](https://www.geeksforgeeks.org/machine-learning/decision-tree/)  
29. A Beginner's Guide to Neural Networks and Deep Learning | Pathmind, accessed August 16, 2025, [http://wiki.pathmind.com/neural-network](http://wiki.pathmind.com/neural-network)  
30. Deciding when to use a Decision Tree Model \- DeepLearning.AI, accessed August 16, 2025, [https://community.deeplearning.ai/t/deciding-when-to-use-a-decision-tree-model/214747](https://community.deeplearning.ai/t/deciding-when-to-use-a-decision-tree-model/214747)  
31. What is a Neural Network? | IBM, accessed August 16, 2025, [https://www.ibm.com/think/topics/neural-networks](https://www.ibm.com/think/topics/neural-networks)  
32. A Neural Network and Overfitting Analogy \- Cow-Shed Startup, accessed August 16, 2025, [https://www.cow-shed.com/blog/a-neural-network-and-overfitting-analogy](https://www.cow-shed.com/blog/a-neural-network-and-overfitting-analogy)  
33. Neural Networks \- A Beginner's Guide \- Towards Data Science, accessed August 16, 2025, [https://towardsdatascience.com/neural-networks-a-beginners-guide-7b374b66441a/](https://towardsdatascience.com/neural-networks-a-beginners-guide-7b374b66441a/)  
34. Feedforward Neural Network \- GeeksforGeeks, accessed August 16, 2025, [https://www.geeksforgeeks.org/nlp/feedforward-neural-network/](https://www.geeksforgeeks.org/nlp/feedforward-neural-network/)  
35. Practical Decision Trees: Real-World Examples and Implementation Tips \- Number Analytics, accessed August 16, 2025, [https://www.numberanalytics.com/blog/practical-decision-trees-real-world-examples-implementation-tips](https://www.numberanalytics.com/blog/practical-decision-trees-real-world-examples-implementation-tips)  
36. Neural Networks vs Decision Trees in Pricing: Pros & Cons \- Pricefx, accessed August 16, 2025, [https://www.pricefx.com/learning-center/neural-networks-vs-decision-trees-in-pricing-pros-cons](https://www.pricefx.com/learning-center/neural-networks-vs-decision-trees-in-pricing-pros-cons)  
37. Why are neural networks described as black-box models? \- Cross Validated, accessed August 16, 2025, [https://stats.stackexchange.com/questions/93705/why-are-neural-networks-described-as-black-box-models](https://stats.stackexchange.com/questions/93705/why-are-neural-networks-described-as-black-box-models)