## dataset
GROOTでファインチューニングできるデータ形式を整理する。
公式Repositoryにはサンプルのデータセットがあるのでそれをリバースエンジニアリングする。

### demo_data
https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/LeRobot_compatible_data_schema.md

```
.
├── data
│   └── chunk-000
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       ├── episode_000002.parquet
│       ├── episode_000003.parquet
│       └── episode_000004.parquet
├── meta
│   ├── episodes.jsonl
│   ├── info.json
│   ├── modality.json
│   ├── stats.json
│   └── tasks.jsonl
└── videos
    └── chunk-000
        └── observation.images.ego_view
            ├── episode_000000.mp4
            ├── episode_000001.mp4
            ├── episode_000002.mp4
            ├── episode_000003.mp4
            └── episode_000004.mp4
```

1. **ビデオ情報**  
episode_00000X.mp4でエピソード毎に動画ファイルを用意する。ここで、`00000X`はエピソード番号。
サンプルは一人称視点でエピソードの時間はきっちり同じでなくてよい。多少バラけている。
次の形式で名前を付ける必要があります:observation.images.<video_name>

    | 項目 | 値 |
    | ---- | ---- |
    | 解像度 | 256 x 256 |
    | フォーマット　| H.264,mp4|
    | FPS | 20|

2. **データ(data/chunk-*)**   
episode_00000X.parquet の命名形式 (X はエピソード番号) に従って、各エピソードに関連付けられたすべての parquet ファイルが含まれます。各 parquet ファイルには次の内容が表構造(parquet)で含まれれる。
データ項目は以下の以下となる

    #### observation.state
    - **内容**: 状態を表す数値の配列
    - **詳細**: modality.json に定義された複数の状態情報が連結されており、現在の環境やエージェントの状態を数値として表現
    
    #### action
    - **内容**: 行動を表す数値の配列
    - **詳細**: modality.json に基づいて、エージェントが実行した行動情報が連結されており、どのような操作や指示が行われたかを数値として記録
    
    #### timestamp
    - **内容**: 観測が取得された時刻
    - **詳細**: この数値は観測のタイミングを示しており、通常は秒などの単位で記載
    
    #### annotation.human.action.task_description
    - **内容**: タスク説明のインデックス番号
    - **詳細**: meta/tasks.jsonl ファイル内に記載されたタスクの説明の中から、該当する説明の番号を示す。以下の例では各データのタスクの概要記載されている`id`を指定。
    
      ```json
      {"task_index": 0, "task": "pick the squash from the counter and place it in the plate"}
      {"task_index": 1, "task": "valid"}
      {"task_index": 2, "task": "pick the ketchup from the counter and place it in the plate"}
      {"task_index": 3, "task": "pick the candle from the counter and place it in the plate"}
      {"task_index": 4, "task": "pick the bar from the counter and place it in the plate"}
      {"task_index": 5, "task": "pick the spray from the counter and place it in the plate"}
      ```
    
    #### task_index
    - **内容**: タスクのインデックス番号です。
    - **詳細**: meta/tasks.jsonl ファイル内のタスクのうち、どのタスクに関連するデータであるかを示す番号になっている。たぶん、`annotation.human.action.task_description`と同じでいいはず。
    
    #### annotation.human.validity
    - **内容**: 人間によるアノテーションの有効性を示す値です。
    - **詳細**: この値が 1 であれば、人間の判断によりデータが「有効」と認められていることを意味します。基本的に手で作成したデータなので`1`でいいはず。
    
    #### episode_index
    - **内容**: エピソードのインデックス番号
    - **詳細**: 同じエピソード内での観測データであることを示す番号になっており、どのエピソードに属しているかを特定できる。
    
    #### index
    - **内容**: データセット全体における観測のグローバルなインデックス番号。
    - **詳細**: データセットでユニークになるような全体でのインデクス番号。
    
    #### next.reward
    - **内容**: 次の観測に対する報酬の値です。
    - **詳細**: 強化学習などで利用される概念で、エージェントの行動の結果として得られる報酬が記録されています。デモデータだと最後だけ1.0になっている。
    
    #### next.done
    - **内容**: エピソードの終了状態を示すブール値です。
    - **詳細**: `false` の場合はエピソードがまだ継続中であることを意味し、`true` であればそのエピソードが終了したことを示す。基本的レコードの最後が`true`

3. **meta data**  

    #### task.json  
    タスクの説明が含まれるファイル

    #### episodes.jsonl  
    エピソードの情報が含まれるる。エピソード長さとtask指示が含まれている。
    ```json
    {"episode_index": 0, "tasks": ["pick the squash from the counter and place it in the plate", "valid"], "length": 416}
    {"episode_index": 1, "tasks": ["pick the ketchup from the counter and place it in the plate", "valid"], "length": 470}
    ```

    #### modality.json​
    状態とアクションのモダリティに関する詳細なメタデータを提供しているファイル。
    具体的には以下の項目がある

    * データの保存と解釈を分離:  
      状態とアクション:連結された float32 配列として保存されます。このmodality.jsonファイルには、これらの配列を、追加のトレーニング情報を含む個別のきめ細かいフィールドとして解釈するために必要なメタデータが含まれれている。

    * ビデオ:  
    個別のファイルとして保存され、構成ファイルにより標準形式に名前を変更できます。注釈:すべての注釈フィールドを追跡します。注釈がない場合は、annotation構成ファイルにフィールドを含めない。
    
    * きめ細かい分割:  
   状態配列とアクション配列を、より意味的に意味のあるフィールドに分割

    * クリア マッピング:  
    データ ディメンションの明示的なマッピング。

    * 高度なデータ変換:  
      トレーニング中にフィールド固有の正規化と回転変換をサポートします。