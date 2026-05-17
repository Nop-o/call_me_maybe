# Cours complet — Projet « Function Calling avec un LLM »

> Ce document couvre toutes les notions nécessaires pour réussir le projet, de zéro.
> Ordre recommandé : lire de bout en bout avant d'écrire la moindre ligne de code.

---

## Table des matières

1. [L'outil `uv` — gestion de projet Python moderne](#1-loutil-uv)
2. [Les LLMs — comment ça marche ?](#2-les-llms)
3. [La tokenisation — du texte aux nombres](#3-la-tokenisation)
4. [Les logits — la sortie brute du modèle](#4-les-logits)
5. [La génération de texte — boucle token par token](#5-la-génération-de-texte)
6. [Le décodage contraint (Constrained Decoding)](#6-le-décodage-contraint)
7. [JSON — rappels et pièges](#7-json)
8. [Le Function Calling — concept clé du projet](#8-le-function-calling)
9. [Architecture du projet — comment tout s'assemble](#9-architecture-du-projet)
10. [Recette pas-à-pas pour l'implémentation](#10-recette-pas-à-pas)

---

## 1. L'outil `uv`

### Qu'est-ce que `uv` ?

`uv` est un gestionnaire de projets Python ultra-rapide (écrit en Rust) qui remplace
la combinaison `pip` + `venv` + `pip-tools`. Il gère :

- la création d'environnements virtuels isolés,
- l'installation de dépendances,
- l'exécution de scripts dans le bon environnement.

### Commandes essentielles

```bash
# Initialiser un projet (crée pyproject.toml + .venv)
uv init mon_projet

# Ajouter une dépendance
uv add torch transformers

# Installer toutes les dépendances déclarées dans pyproject.toml
uv sync

# Lancer un script Python DANS l'environnement du projet
uv run python -m src

# Lancer avec des arguments (comme demandé dans le projet)
uv run python -m src --input data/input/tests.json --output data/output/results.json
```

### Structure type d'un projet uv

```
mon_projet/
├── pyproject.toml        ← déclaration du projet et de ses dépendances
├── uv.lock               ← versions exactes verrouillées (ne pas modifier à la main)
├── .venv/                ← environnement virtuel (ne pas versionner)
├── src/
│   ├── __init__.py
│   └── __main__.py       ← point d'entrée pour "python -m src"
└── data/
    ├── input/
    └── output/
```

### Le fichier `pyproject.toml`

```toml
[project]
name = "call-me-maybe"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch",
    "transformers",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### Pourquoi `python -m src` ?

Lancer `python -m src` demande à Python d'exécuter le module `src` comme un
programme. Python cherche alors `src/__main__.py` et l'exécute. C'est le point
d'entrée standard pour les projets structurés en package.

---

## 2. Les LLMs

### Définition

Un **Large Language Model (LLM)** est un réseau de neurones entraîné pour prédire
le prochain mot (token) dans une séquence de texte. Il n'a pas de logique symbolique,
pas de calculatrice interne — il prédit des probabilités sur des tokens.

### Ce qu'un LLM fait vraiment

```
Entrée  : "Quelle est la somme de 2 et"
Sortie  : distribution de probabilité sur tous les tokens connus
          → "3"   : 12 %
          → " 3"  : 45 %
          → " deux" : 8 %
          → ...
```

Le modèle choisit ensuite le token le plus probable (ou en échantillonne un), l'ajoute
à la séquence, et recommence.

### Paramètres du modèle

Un modèle à **0,5 milliard de paramètres** (comme Qwen3-0.6B utilisé ici) est
relativement petit. Sans guidage structurel, il peut produire du JSON mal formé. C'est
précisément pourquoi le décodage contraint est indispensable.

### Tenseurs et PyTorch (notions de base)

Le SDK fourni utilise PyTorch. Un **tenseur** est simplement un tableau
multidimensionnel (comme un `numpy.ndarray`).

```python
import torch

# Vecteur 1D (liste de token IDs)
input_ids = torch.tensor([[892, 318, 262, 4771]])  # shape: [1, 4]

# Obtenir un élément
input_ids[0][2]  # → tensor(262)
```

---

## 3. La tokenisation

### Pourquoi tokeniser ?

Les réseaux de neurones ne traitent que des nombres. La tokenisation convertit
le texte en une séquence d'entiers que le modèle comprend.

### Un token ≠ un mot

Les tokeniseurs modernes (BPE — *Byte Pair Encoding*, SentencePiece, etc.) découpent
le texte en **sous-mots**. Exemples :

```
"hello"       → ["hello"]                         → [15339]
"unhappiness" → ["un", "happiness"]               → [665, 20095]
"fn_add"      → ["fn", "_", "add"]                → [3944, 62, 723]
```

Les espaces sont souvent encodés comme partie du token suivant (ex. `" the"` est
un token différent de `"the"`).

### Ce que le SDK vous donne

```python
model = Small_LLM_Model()

# Texte → liste d'entiers
ids = model.encode("What is the sum?")
# → [3838, 374, 279, 2790, 30]

# Entiers → texte (optionnel)
text = model.decode([3838, 374, 279, 2790, 30])
# → "What is the sum?"

# Chemin vers le vocabulaire JSON
vocab_path = model.get_path_to_vocabulary_json()
```

### Le fichier vocabulaire JSON

Ce fichier est **crucial** pour le décodage contraint. Il contient la correspondance
entre chaque `token_id` (entier) et sa représentation en chaîne de caractères.

```json
{
  "0": "<|endoftext|>",
  "1": "!",
  "2": "\"",
  "220": " ",
  "1881": "true",
  "3204": "false",
  "90": "{",
  "92": "}",
  ...
}
```

Chargez-le une seule fois au démarrage :

```python
import json

with open(model.get_path_to_vocabulary_json()) as f:
    vocab = json.load(f)  # { "0": "<|endoftext|>", "1": "!", ... }

# Inversez-le pour chercher un token_id à partir de sa chaîne
token_to_id = {v: int(k) for k, v in vocab.items()}
```

---

## 4. Les logits

### Définition

Après avoir traité les `input_ids`, le modèle retourne un **tenseur de logits** :
un score brut (non normalisé) pour chaque token du vocabulaire.

```python
logits = model.get_logits_from_input_ids(input_ids)
# shape : [1, sequence_length, vocab_size]
# Ex.  : [1, 9, 32000]  → 32 000 tokens possibles
```

### Logits → probabilités

On applique une **softmax** pour transformer les scores en probabilités :

```
P(token_i) = exp(logit_i) / Σ exp(logit_j)
```

En pratique, pour sélectionner le prochain token, on prend uniquement les logits
de la **dernière position** de la séquence :

```python
next_token_logits = logits[0, -1, :]  # shape: [vocab_size]
```

### Sélection du token le plus probable (greedy)

```python
import torch

next_token_id = torch.argmax(next_token_logits).item()
```

### Pourquoi travailler avec des logits et pas des probabilités ?

Parce que mettre un score à **−∞** (`float('-inf')`) avant la softmax rend la
probabilité de ce token exactement **0**, sans affecter les autres. C'est le
mécanisme du décodage contraint.

---

## 5. La génération de texte

### La boucle de génération

```
┌─────────────────────────────────────────────────────────┐
│  Prompt initial : "What is the sum of 2 and 3?"         │
│                                                          │
│  Étape 1 → encode → input_ids → LLM → logits            │
│            → choisir next_token → ajouter aux ids        │
│                                                          │
│  Étape 2 → [prompt + token_1] → LLM → logits            │
│            → choisir next_token → ajouter aux ids        │
│                                                          │
│  ... (répéter jusqu'à token de fin ou longueur max)      │
└─────────────────────────────────────────────────────────┘
```

### Implémentation de base (sans contrainte)

```python
import torch

def generate(model, prompt: str, max_new_tokens: int = 50) -> str:
    # 1. Encoder le prompt
    input_ids = model.encode(prompt)
    generated_ids = list(input_ids)

    for _ in range(max_new_tokens):
        # 2. Obtenir les logits
        ids_tensor = torch.tensor([generated_ids])
        logits = model.get_logits_from_input_ids(ids_tensor)

        # 3. Prendre les logits du dernier token
        next_logits = logits[0, -1, :]

        # 4. Choisir le token suivant (greedy)
        next_id = torch.argmax(next_logits).item()

        # 5. Condition d'arrêt (token de fin de séquence)
        if next_id == EOS_TOKEN_ID:
            break

        # 6. Ajouter le token et recommencer
        generated_ids.append(next_id)

    return model.decode(generated_ids[len(input_ids):])
```

---

## 6. Le décodage contraint

C'est **le cœur du projet**. L'objectif : à chaque étape de génération, n'autoriser
que les tokens qui maintiennent un JSON valide et conforme au schéma attendu.

### Principe

```
Logits bruts du modèle
        ↓
Masque des tokens invalides (→ -inf)
        ↓
Logits filtrés (seuls les tokens valides ont un score > -inf)
        ↓
argmax / sample → token forcément valide
```

### Structure JSON à générer

Pour chaque prompt, vous devez produire exactement ceci :

```json
{"name": "fn_add_numbers", "parameters": {"a": 40.0, "b": 2.0}}
```

Cette structure a une **grammaire fixe** que vous pouvez exploiter.

### Automate d'états pour guider la génération

L'idée est de modéliser les "états" dans lesquels se trouve votre JSON en cours
de génération. À chaque état, seuls certains tokens sont valides.

```
ÉTAT 0 : début              → seul token valide : "{"
ÉTAT 1 : après "{"          → seul token valide : '"name"'
ÉTAT 2 : après '"name"'     → seul token valide : ":"
ÉTAT 3 : après ":"          → tokens valides : l'un des noms de fonctions (entre guillemets)
ÉTAT 4 : après le nom       → seul token valide : ","
ÉTAT 5 : après ","          → seul token valide : '"parameters"'
ÉTAT 6 : ...
```

### Approche pratique : génération caractère par caractère via tokens

Concrètement, vous allez construire le JSON **token par token**, en tenant compte
de ce qui a déjà été généré pour décider quels tokens sont autorisés ensuite.

Une approche robuste consiste à travailler avec une **machine à états finis** dont
les transitions dépendent du contenu JSON déjà produit.

### Exemple concret : restreindre le nom de la fonction

```python
def get_valid_function_name_tokens(vocab: dict, function_names: list[str]) -> set[int]:
    """
    Retourne les token IDs correspondant aux débuts valides
    d'un nom de fonction entre guillemets.
    """
    valid_ids = set()
    for token_id, token_str in vocab.items():
        for name in function_names:
            quoted = f'"{name}"'
            if quoted.startswith(token_str) or token_str.startswith('"'):
                valid_ids.add(int(token_id))
    return valid_ids
```

### Appliquer le masque

```python
def apply_mask(logits: torch.Tensor, valid_ids: set[int]) -> torch.Tensor:
    """Met à -inf tous les tokens non autorisés."""
    mask = torch.full_like(logits, float('-inf'))
    for token_id in valid_ids:
        mask[token_id] = logits[token_id]
    return mask
```

### Cas particulier : les nombres

Les nombres JSON peuvent être multi-tokens (`"40"` peut être tokenisé en `["4", "0"]`
ou `["40"]`). Votre automate doit autoriser :

- les chiffres `0-9`
- le point décimal `.`
- le signe `-` (nombres négatifs)
- le passage au token suivant (`,` ou `}`) quand un nombre complet est formé

### Cas particulier : les chaînes de caractères

Une valeur string en JSON est délimitée par `"..."`. Vous devez :

1. Autoriser le guillemet ouvrant `"`
2. Autoriser n'importe quel caractère de contenu (sauf `"` non échappé)
3. Autoriser le guillemet fermant `"`

---

## 7. JSON

### Rappels de syntaxe

```json
{
  "name": "fn_add_numbers",
  "parameters": {
    "a": 40.0,
    "b": 2.0
  }
}
```

Règles strictes :
- Les clés sont **toujours** entre guillemets doubles
- Pas de virgule après le dernier élément
- Pas de commentaires
- Les nombres peuvent être entiers (`2`) ou flottants (`2.0`)
- Les booléens sont `true` / `false` (minuscules)

### Lecture/écriture en Python

```python
import json

# Lire un fichier JSON
with open("data/input/functions_definition.json") as f:
    functions = json.load(f)

# Écrire un fichier JSON
results = [{"prompt": "...", "name": "fn_add", "parameters": {"a": 2.0}}]
with open("data/output/results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# Gestion d'erreur (fichier manquant ou JSON invalide)
try:
    with open("data/input/tests.json") as f:
        tests = json.load(f)
except FileNotFoundError:
    print("Fichier introuvable")
    tests = []
except json.JSONDecodeError as e:
    print(f"JSON invalide : {e}")
    tests = []
```

### Correspondance de types JSON ↔ Python

| JSON       | Python      |
|------------|-------------|
| `number`   | `float`     |
| `string`   | `str`       |
| `boolean`  | `bool`      |
| `null`     | `None`      |
| `array`    | `list`      |
| `object`   | `dict`      |

---

## 8. Le Function Calling

### Concept

Le **function calling** est la capacité d'un LLM à identifier, pour une requête
en langage naturel, quelle fonction appeler et avec quels arguments.

L'idée est de transformer :

```
"What is the sum of 40 and 2?"
```

en une **représentation structurée** :

```json
{
  "name": "fn_add_numbers",
  "parameters": {"a": 40.0, "b": 2.0}
}
```

Le modèle n'exécute pas la fonction — il produit uniquement la description de l'appel.

### Utilisation dans le projet

Les fonctions disponibles sont décrites dans `functions_definition.json`. Votre
programme doit :

1. Lire les définitions de fonctions
2. Pour chaque prompt, déterminer quelle fonction appelle le prompt
3. Extraire les arguments avec les bons types
4. Produire le JSON de résultat

### Construire le prompt système

Pour guider le LLM, vous allez lui fournir un **prompt** qui décrit les fonctions
disponibles et lui demande de répondre uniquement en JSON :

```python
def build_prompt(user_query: str, functions: list[dict]) -> str:
    functions_desc = json.dumps(functions, indent=2)
    return f"""You are a function calling assistant.
Available functions:
{functions_desc}

User request: {user_query}

Respond ONLY with a JSON object with keys "name" and "parameters".
JSON:"""
```

**Attention** : comme précisé dans le sujet, compter uniquement sur le prompt
sans décodage contraint n'est pas fiable. Le décodage contraint est obligatoire.

---

## 9. Architecture du projet

### Vue d'ensemble recommandée

```
src/
├── __init__.py
├── __main__.py          ← parse les args CLI, orchestre tout
├── loader.py            ← lit les fichiers JSON d'entrée
├── prompt_builder.py    ← construit le prompt à envoyer au LLM
├── constrained_decoder.py ← cœur : génération token par token avec masque
├── schema_validator.py  ← vérifie les types des paramètres en sortie
└── writer.py            ← écrit le JSON de sortie
```

### Le `__main__.py`

```python
import argparse
import json
from .loader import load_functions, load_tests
from .constrained_decoder import decode_function_call
from .writer import write_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--functions_definition", default="data/input/functions_definition.json")
    parser.add_argument("--input", default="data/input/function_calling_tests.json")
    parser.add_argument("--output", default="data/output/function_calling_results.json")
    args = parser.parse_args()

    functions = load_functions(args.functions_definition)
    tests = load_tests(args.input)

    results = []
    for test in tests:
        result = decode_function_call(test["prompt"], functions)
        results.append(result)

    write_results(results, args.output)

if __name__ == "__main__":
    main()
```

### Flux de données complet

```
functions_definition.json ──┐
                             ├→ build_prompt(query, functions)
tests.json ─→ prompt ────────┘
                  │
                  ↓
           model.encode(prompt) → input_ids
                  │
                  ↓
    ┌─────────────────────────────────────────┐
    │         BOUCLE DE GÉNÉRATION            │
    │                                         │
    │  input_ids → get_logits → next_logits   │
    │                  ↓                      │
    │         apply_constraint_mask           │
    │                  ↓                      │
    │          argmax → next_token_id         │
    │                  ↓                      │
    │     append to input_ids & generated     │
    │                                         │
    │     si JSON complet → sortir            │
    └─────────────────────────────────────────┘
                  │
                  ↓
          json.loads(generated_text)
                  │
                  ↓
      {"name": "...", "parameters": {...}}
                  │
                  ↓
       function_calling_results.json
```

---

## 10. Recette pas-à-pas pour l'implémentation

Voici l'ordre recommandé pour implémenter le projet sans se perdre.

### Étape 1 — Mettre en place l'environnement

```bash
uv sync          # installe les dépendances
uv run python -c "from llm_sdk import Small_LLM_Model; print('OK')"
```

### Étape 2 — Explorer le vocabulaire

```python
import json
from llm_sdk import Small_LLM_Model

model = Small_LLM_Model()
vocab = json.load(open(model.get_path_to_vocabulary_json()))

# Trouver le token ID de "{"
for k, v in vocab.items():
    if v == "{":
        print(f'token "{v}" → id {k}')
```

### Étape 3 — Générer sans contrainte (baseline)

Faire une génération simple pour voir ce que le modèle produit naturellement.
Cela aide à calibrer le prompt et à comprendre les tokens générés.

### Étape 4 — Implémenter l'automate d'états

Définir les états de votre JSON en cours de construction et les tokens autorisés
dans chaque état. Commencer par un JSON simple (1 seul paramètre de type number).

### Étape 5 — Gérer les types

Étendre l'automate pour gérer `string`, `number`, `boolean`. Chaque type implique
des règles de tokens autorisés différentes.

### Étape 6 — Gérer plusieurs paramètres

Permettre la génération de plusieurs paires `"clé": valeur` séparées par des virgules.

### Étape 7 — Tester sur les exemples fournis

Vérifier que le output JSON passe un `json.loads()` sans erreur et que les types
correspondent aux définitions.

### Étape 8 — Gestion robuste des erreurs

- Fichier JSON manquant ou invalide
- Prompt ambigu (aucune fonction ne correspond clairement)
- Paramètres avec des valeurs inhabituelles (nombres négatifs, chaînes vides)

---

## Glossaire rapide

| Terme | Définition |
|---|---|
| **Token** | Unité de texte traitée par le LLM (sous-mot, symbole, etc.) |
| **Token ID** | Identifiant entier d'un token dans le vocabulaire |
| **Logits** | Scores bruts (non normalisés) produits par le LLM pour chaque token |
| **Softmax** | Fonction qui convertit des logits en probabilités (somme = 1) |
| **Greedy decoding** | Toujours choisir le token de plus haute probabilité |
| **Constrained decoding** | Masquer les tokens invalides avant la sélection |
| **BPE** | Byte Pair Encoding — algorithme de tokenisation courant |
| **Vocabulaire** | Ensemble de tous les tokens connus du modèle |
| **Function calling** | Capacité d'un LLM à identifier et formater un appel de fonction |
| **Schema** | Structure attendue d'un JSON (clés, types, obligatoire/optionnel) |
| **uv** | Gestionnaire de projets Python rapide (remplace pip + venv) |
| **Tensor** | Tableau multidimensionnel utilisé par PyTorch |

---

*Bonne chance pour le projet !*