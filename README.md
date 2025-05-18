# Hybrid Neural Solver for ODEs

This repository contains the complete implementation used in the article *“Analysis of the Generalization Capacity of Neural Networks for Solving ODEs via Hybrid RK4-Supervised Training”* by Julia Cohen.

## 📘 Description

The proposed approach combines the fourth-order Runge-Kutta method (RK4) for supervised data generation with multilayer perceptron-based artificial neural networks (ANNs). The methodology is applied to the Verhulst logistic equation, including scenarios with Gaussian noise, and evaluates how different training intervals affect the model's extrapolation capability.

Each configuration ([0,1], [0,2], [0,4], [0,6]) was tested with three fixed seeds (0, 1, 2) and one random initialization, allowing a statistical comparison of the network's robustness.

## 📁 Repository Structure

- `verhulst_treino_0_a_X_seedY.py`: Python scripts for each training interval and seed.
- `results/`: Directory containing all generated figures (predicted vs analytical curves).

## 🔧 Requirements

- Python 3.12  
- PyTorch  
- Matplotlib  
- *Note:* NumPy is not required. All operations are performed using PyTorch tensors.

---

Developed with 💻 by Julia Cohen



---

## ✅ `README.md` – Versão em Português

# Solução Híbrida com Redes Neurais para EDOs

Este repositório contém a implementação completa utilizada no artigo *“Análise da Capacidade de Generalização de Redes Neurais na Solução de EDOs via Treinamento Híbrido com RK4”*, desenvolvido por Julia Cohen.

## 📘 Descrição

A proposta combina o método de Runge-Kutta de quarta ordem (RK4) para geração de dados supervisionados com redes neurais artificiais do tipo perceptron multicamada. A metodologia é aplicada à equação logística de Verhulst, incluindo casos com ruído gaussiano, e avalia como diferentes intervalos de treinamento impactam a capacidade de extrapolação do modelo.

Cada configuração ([0,1], [0,2], [0,4], [0,6]) foi testada com três sementes fixas (0, 1, 2) e uma inicialização aleatória, permitindo uma análise estatística da robustez da rede.

## 📁 Estrutura do Repositório

- `verhulst_treino_0_a_X_seedY.py`: Scripts Python para cada intervalo e semente.
- `results/`: Pasta com os gráficos gerados (previsão vs solução analítica).

## 🔧 Requisitos

- Python 3.12  
- PyTorch  
- Matplotlib  
- *Observação:* Não é necessário instalar o NumPy. Todas as operações são feitas com tensores do PyTorch.

---

Feito com 💻 por Julia Cohen
