open Rune
open Kaun

let load_names path = Saga.read_lines path |> List.map String.lowercase_ascii

let build_vocab (names : string list) =
  let tbl = Hashtbl.create 64 in
  Hashtbl.add tbl '.' 0;
  List.iter
    (fun n ->
      String.iter
        (fun c ->
          if not (Hashtbl.mem tbl c) then Hashtbl.add tbl c (Hashtbl.length tbl))
        n)
    names;
  let vocab_size = Hashtbl.length tbl in
  let idx2ch = Array.make vocab_size '.' in
  Hashtbl.iter (fun c i -> if i < vocab_size then idx2ch.(i) <- c) tbl;
  let ch2idx c = match Hashtbl.find_opt tbl c with Some i -> i | None -> 0 in
  let encode (s : string) =
    s |> String.to_seq |> List.of_seq |> List.map ch2idx
  in
  let decode ids =
    ids |> List.map (fun i -> idx2ch.(i)) |> List.to_seq |> String.of_seq
  in
  (vocab_size, ch2idx, encode, decode)

let make_dataset ~block_size ~batch_size ~vocab_size:_ ~encode names =
  let tokenize s = encode ("." ^ s ^ ".") in
  Dataset.sliding_window ~block_size ~tokenize ~device:c names
  |> Dataset.batch batch_size
  |> Dataset.shuffle ~buffer_size:200

let split_names_for_eval names =
  let n = List.length names in
  let test_set_size = Stdlib.min 1000 (Stdlib.max 1 (n / 10)) in
  let indices = Array.init n Fun.id in
  let st = Random.State.make [| 3407 |] in
  for i = n - 1 downto 1 do
    let j = Random.State.int st (i + 1) in
    let tmp = indices.(i) in
    indices.(i) <- indices.(j);
    indices.(j) <- tmp
  done;
  let train_idx = Array.sub indices 0 (n - test_set_size) in
  let test_idx = Array.sub indices (n - test_set_size) test_set_size in
  let arr = Array.of_list names in
  let train = Array.to_list (Array.map (Array.get arr) train_idx) in
  let test = Array.to_list (Array.map (Array.get arr) test_idx) in
  (train, test)

let make_optimizer ~lr ~weight_decay =
  if weight_decay > 0.0 then Optimizer.adamw ~lr ~weight_decay ()
  else Optimizer.adam ~lr ()

let train_mlp ~vocab_size ~block_size ~n_embd ~n_embd2 ~epochs ~lr ~weight_decay
    ~val_data train_data =
  let open Layer in
  let model =
    sequential
      [
        embedding ~vocab_size ~embed_dim:n_embd ();
        flatten ();
        linear ~in_features:(block_size * n_embd) ~out_features:n_embd2 ();
        tanh ();
        linear ~in_features:n_embd2 ~out_features:vocab_size ();
      ]
  in
  let optimizer = make_optimizer ~lr ~weight_decay in
  let state, _ =
    Training.fit ~model ~optimizer
      ~loss_fn:Loss.softmax_cross_entropy_with_indices ~train_data ~val_data
      ~epochs ~progress:true ~rngs:(Rng.key 42) ~device:c ~dtype:float32 ()
  in
  (model, state)

let () =
  let names = load_names "names.txt" in
  let vocab_size, _ch2idx, encode, _decode = build_vocab names in
  let train_names, eval_names = split_names_for_eval names in
  let bs = 16 and batch_size = 256 in
  let train_ds =
    make_dataset ~block_size:bs ~batch_size ~vocab_size ~encode train_names
  in
  let eval_ds =
    make_dataset ~block_size:bs ~batch_size ~vocab_size ~encode eval_names
  in
  let _model, state =
    train_mlp ~vocab_size ~block_size:bs ~n_embd:64 ~n_embd2:64 ~epochs:1
      ~lr:2e-3 ~weight_decay:0.01 ~val_data:eval_ds train_ds
  in
  Printf.printf "[mlp] Training complete.\n%!";
  (* Reset eval dataset iterator; it was consumed during validation inside
     fit *)
  Dataset.reset eval_ds;
  let eval_nll, _ =
    Training.evaluate ~state ~dataset:eval_ds
      ~loss_fn:Loss.softmax_cross_entropy_with_indices ()
  in
  let ppl = Stdlib.exp eval_nll in
  Printf.printf "[mlp] Eval NLL: %.4f nats, PPL: %.2f\n%!" eval_nll ppl
