---
layout: post
title: "[E] Graph Neural Networks"
excerpt: "Graph Neural Networks (GNNs) is currently the primer approach in applying neural networks to relational data. In this post, we will see the basic formulation and different variations of the GNNs."
published: false
comments: true
mathjax: true
---

# Graph Neural Networks

> Graph Neural Networks (GNNs) is currently the primer approach in applying neural networks to relational data. In this post, we will see the basic formulation and different variations of the GNNs.

* [A Brief Introduction to Graphs](#A-Brief-Introduction-to-Graphs)
* [Machine Learning on graphs](#Machine-Learning-on-graphs)
    * [Node Classification/Regression](#1.-Node-Classification/Regression)
    * [Link Prediction](#2.-Link-Prediction)
    * [Community Detection & Clustering](#3.-Community-Detection-&-Clustering)
    * [Graph Classification/Regression](#4.-Graph-Classification/Regression)
* [Simple Naive Approach (Adjacency Matrix)](#Simple-Naive-Approach-(Adjacency-Matrix))
* [Graph Neural Networks (Formulation)](#Graph-Neural-Networks-(Formulation))
    * [Neural Message Passing](#Neural-Message-Passing)
* [Examples](#Examples)
* [Conclusion](#Conclusion)

## A Brief Introduction to Graphs

Graphs are structures that can store complex relations (edges) between objects (nodes). These relations can happen in many shapes. For instance, friendships in the social networks, [protein-protein interactions (PPIs)](https://en.wikipedia.org/wiki/Protein%E2%80%93protein_interaction), or the user-product interaction as in the recommender systems. We can define all of these problems as a graph and answer lots of questions. For example, what should we recommend to some particular user, what is the role of specific proteins, or which users are fake (bots) in social media? 

To make things a little bit more concrete, let's start by defining the graphs formally and explain some properties related to them.

> Graphs are also called networks, but we try to restrict ourselves to use the term graph to prevent the potential confusion between the neural networks.

Graphs $G = (V, E)$ include set of nodes $(V)$ and set of edges $(E)$. The connection between the nodes is called edges. We can represent the graph as an adjacency matrix $(A)$ where every cell in the matrix is either 0 or 1. If there is a weight between the nodes the adjacency matrix can take arbitrary numbers as well. We can also represent the graphs via the edge list. In this type of representation, we only store the edges between the nodes. Here is an example of a toy graph with only 6 nodes and 7 (undirected) edges.

<img src="first_graph.PNG" alt="drawing" width="200"/>

> **Adjacency matrix:**

| |A|B|C|D|E|F|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|**A**|1|1|0|0|0|0|
|**B**|1|1|1|1|0|0|
|**C**|0|1|1|1|0|0|
|**D**|0|1|1|1|1|1|
|**E**|0|0|0|1|1|1|
|**F**|0|0|0|1|1|1|

> **Edge list**: [(A, B), (B, C), (B, D), (C, D), (D, E), (D, F), (E, F)]

Graphs can be directed/undirected depending on the problem. For example, the relations on the Facebook social graph are undirected. If you add someone as a friend, you become a friend mutually. On the other side, if you follow someone on Twitter, the relation is directed. You don't follow each other mutually. 

Apart from being directed/undirected, graphs can also have different types of nodes which we call heterogeneous graphs. For example in the recommender systems, we can represent the users and the items (e.g. movies or products) as separate nodes.

## Machine Learning on graphs

There are multiple types of problem formulations in the graph domain that can be solved with supervised/unsupervised settings similar to classical machine learning. Here we will touch upon four of them.

#### 1. **Node Classification/Regression**

In this type of problem, we aim to solve the node-level tasks such as the detection of fake users (bots) in social media. Bots tend to follow similar patterns in terms of actions (following/posting). We represent the users as nodes in the graph and learn node embeddings based on the graph structure and input features like follower count and post frequency. Then we classify the users based on the node embeddings.

#### 2. **Link Prediction**

Link prediction is a task to understand the potential relations between the nodes. For instance, in the recommender systems, we can represent the users and the products as nodes in the graph. We can add edges based on the user purchases and define the problem as a link prediction task. At test time, we can make predictions using the learned user and product embeddings and recommend products to users based on the prediction results such as 0/1. [KNOWLEDGE GRAPH EXAMPLE HERE (!!!)]

#### 3. **Community Detection & Clustering**

In community detection & clustering problems, we aim to group similar nodes. For example, in citation graphs, we can represent papers as nodes and create edges between the nodes based on citations. After learning the unsupervised representation of nodes, we can apply any clustering algorithm like KMeans to generate clusters/groups.

#### 4. **Graph Classification/Regression**

Graph-based tasks are similar to node-level tasks but require making predictions about the full graph instead of just a single node or link. For example, understanding the [threads in the Reddit](https://snap.stanford.edu/data/reddit_threads.html) is a discussion-based or not is a binary graph-classification task. Here the nodes are the users, and the links are the replies between them. To make predictions on the graph, it is also essential to incorporate the global structure together with the local features.

After discussing the several problems related to graphs, let's see how we can apply the neural networks to the graph data.

#### Simple Naive Approach (Adjacency Matrix)

One basic idea to use the neural networks with the graph-structured data is to use the adjacency matrix as an input. We can flatten the adjacency matrix to be a 1d vector and supply it to the network. The major drawback of this approach is that the adjacency matrix is not permutation invariant. Different ordering of the nodes changes the adjacency matrix and hence the input to the neural network. We will have different outputs for the same graph by just reordering the adjacency matrix. This is not desirable since the input graph is essentially the same. So, one main property that we need to satisfy is the permutation invariance when applying neural networks.

We saw that the adjacency matrix as an input is not sufficient. What can we do? It is time for Graph Neural Networks to come into the stage.


## Graph Neural Networks (Formulation)

![](gnn_computation_graph.png)

The motivating idea behind the GNNs is applying the neural networks to the graphs to extract useful features. The fundamental principle is the exchange information between the nodes. This communication is called **message passing** and, updating the node information with neural networks is called **neural message passing**.

There are two key components in the GNNs framework:

* **Aggregation**: Aggregating the information from the neighbors.
* **Update**: Updating the node information based on the previous information embedding and aggregated message.

#### Neural Message Passing

As we stated earlier, neural message passing is the process of exchanging embedding messages between the nodes and updating with the neural networks. Every message passing layer aims to **aggregate** the neighbor's information and create a message vector. 

Here is the abstract definition:
$$
\begin{align}
h_{u}^{(k+1)} &= UPDATE^{(k)}\bigg(h_{u}^{(k)}, AGGREGATE^{(k)}(\{h_v^{(k)}, \forall{v} \in \mathcal{N}(u)\})\bigg) \\
&= UPDATE^{(k)} \bigg(h_{u}^{(k)}, m_{\mathcal{N}(u)}^{(k)}\bigg)
\end{align}
$$

In this equation, $m_{\mathcal{N}(u)}^{(k)}$ is defined as $ m_{\mathcal{N}(u)}^{(k)} = AGGREGATE^{(k)}(\{h_v^{(k)}, \forall{v} \in \mathcal{N}(u)\})$. The aggregation function here can be a non-parametric function such as mean or sum or more complex such as neural networks. To make the computations a bit of concrete, let's convert them to the equations which we can implement.

$$
h_{u}^{(k+1)} = \sigma\bigg(W_{\text{self}}^{(k)}h_{u}^{(k)} + W_{\text{neigh}}^{(k)}\sum_{\substack{v \in \mathcal{N} (u)}}h_v^{(k)} + b^{(k)}\bigg)
$$

Here $\sigma$ is an element-wise activation function (e.g. ReLU or Tanh) and $W_{\text{self}}$ and $W_{\text{neigh}}$ are learnable parameters. As you might notice, we used the $\sum$ operator as an aggregation function. If we want to map this equation to our abstract definition:

$$
m_{\mathcal{N}(u)} = \sum_{\substack{v \in \mathcal{N} (u)}}h_v
$$
$$
\text{UPDATE}(h_{u}, m_{\mathcal{N}(u)}) = \sigma\bigg(W_{\text{self}}h_{u} + W_{\text{neigh}}m_{\mathcal{N}(u)} + b\bigg)
$$

Here we can skip the update operation by adding self-loops to the input graph. In this case, the final equation would be

$$h_{u}^{(k+1)} = \text{AGGREGATE}(\{h_v^{(k)}, \forall_{v} \in \mathcal{N}(u) \cup {\{u\}}\})$$

like this. Aggregation functions should satisfy an important property: being permutation invariant. The nodes in the graph have no particular order. The aggregation function should produce always the same result independent from the input order. The mean and the sum function obey this rule. However, we should pay attention when we consider the architectures which have learnable parameters.

It is possible to use the LSTMs as an aggregator function. As you know, LSTMs process the input sequentially. Although having more representational capability brings an advantage, our rule of being "permutation invariant" is unfortunately broken.  To overcome this problem, the neighbors of the nodes were given randomly in different orders to the LSTM network. Our goal here is to learn representations independent of the order. Lastly, the final message information is obtained by averaging the outputs.

It is also possible to use **attention** as another aggregation function. As you might guess, not all the neighbors of a node have equal importance. For example, in citation graphs, some papers may have been cited from much more various articles. They may contain less valuable information when determining the class of a document (as it gets more citations). By applying attention, the importance of the nodes is learned. While any attention mechanism in the literature is possible to use, we will consider the self-attention as in [Veličković et al.](https://arxiv.org/abs/1710.10903) Firstly, we compute the attention between the node $i$ and $j$.

$$
\begin{equation} 
e_{ij} = a({\bf W}\vec{h}_i, {\bf W}\vec{h}_j)
\end{equation}
$$

To convert them to probabilities, let's apply the softmax function: 

$$
\alpha_{ij} = \mathrm{softmax}_j(e_{ij})=\frac{\exp(e_{ij})}{\sum_{k\in\mathcal{N}_i} \exp(e_{ik})}
$$

and finally let's apply a nonlinear activation function:

$$
\begin{equation}\label{eqnatt}
	\vec{h}'_i = \sigma\left(\sum_{j\in\mathcal{N}_i} \alpha_{ij} {\bf W}\vec{h}_j\right).
\end{equation}
$$

The self-attention operation here is applied to the full graph. The authors proposed to inject the graph structure in to the mechanism by performing masked-attention. So, only the attention on edge $e_{ij}$ considered where $j$ is in the neighborhood of node $i$.

Additionally, as in [Vaswani et al.,](https://arxiv.org/abs/1706.03762) we can use multi-head attention as well. By applying multiple attention heads independent from each other, we increase the representation power and stabilize the training.

## Examples

The complete code example for this blog can be found on this repository. Here we will focus only the output image. In the below graph, we use the [Cora] dataset as our prediction task. Cora dataset includes 2708 papers from 


![](attention_graph.png)


## Conclusion



```python

```
