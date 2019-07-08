
- 200x200の画像に黒で壁を書いて、青でエサの点を書いたものをmakemeshに与えるとメッシュを作ります
- そのとき作ったメッシュが非連結だと動きません
    - 通路が細すぎるとメッシュがちぎれます

# requirements
- numpy
- matplotlib (imagemagick or pillow enabled)
- cv2
- meshzoo (https://pypi.org/project/meshzoo/)

# examples

## kanto
```
python makemesh.py kanto.png --output kanto_node.pickle
python makeAnimation.py kanto.png kanto_node.pickle --output kanto_anime.gif
```

## maze
```
python makemesh.py maze.png --output maze_node.pickle
python makeAnimation.py maze.png maze_node.pickle --output maze_anime.gif --frames 20
```
