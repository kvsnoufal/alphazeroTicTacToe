# Mastering TicTacToe with AlphaZero

Coding AlphaZero algorithm from scratch to play the game of TicTacToe and it never loses!!


Pls checkout the [medium article](https://medium.com/@noufalsamsudin/mastering-tictactoe-with-alphazero-cc28998bf36c) for a quick overview.

Demo: https://alphazerotictactoe.herokuapp.com/

To build training dataset using self play:
```
python selfplay.py
```

To train the value and policy networks
```
python retrain.py
```

AlphaZero (or it's more famous predecessor AlphaGo) made one of the most famous breakthroughs in the field of AI. Being able to achieve superhuman performance in the games of chess, shogi and go, having a Netflix movie made about its accomplishment (AlphaGo - The Movie) are just some of its accolades.

In this article, I will be discussing the general intuition behind AlphaZero and explaining the various components and processes in the algorithm. I have modified the algorithm from the paper to play TicTacToe.


![Pic of results](https://github.com/kvsnoufal/alphazeroTicTacToe/blob/main/doc/1.img.gif)
![Pic of results2](https://github.com/kvsnoufal/alphazeroTicTacToe/blob/main/doc/2.img.gif)

Demo: https://alphazerotictactoe.herokuapp.com/



## Shoulders of giants
1. Mastering the game of Go without human knowledge: https://www.nature.com/articles/nature24270
2. http://joshvarty.github.io/AlphaZero/
3. AlphaZero Cheatsheet: https://medium.com/applied-data-science/alphago-zero-explained-in-one-diagram-365f5abf67e0
4. A Simple Alpha(Go) Zero Tutorial: https://web.stanford.edu/~surag/posts/alphazero.html
5. https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
6. https://www.youtube.com/watch?v=MPXGiowUr0o&ab_channel=SkowstertheGeek
