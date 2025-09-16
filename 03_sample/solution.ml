open Rune
open Kaun

(* 01 - Dataset
   -------------------------------------------------------------- *)

let load_names path = Saga.read_lines path |> List.map String.lowercase_ascii

type vocab = {
  size : int;
  ch2idx : char -> int;
  encode : string -> int list;
  decode : int list -> string;
}

(* Build a simple character vocabulary with '.' as a boundary token. *)
let build_vocab (names : string list) : vocab =
  let tbl = Hashtbl.create 64 in
  Hashtbl.add tbl '.' 0;
  List.iter
    (fun n ->
      String.iter
        (fun c ->
          if not (Hashtbl.mem tbl c) then Hashtbl.add tbl c (Hashtbl.length tbl))
        n)
    names;
  let size = Hashtbl.length tbl in
  let idx2ch = Array.make size '.' in
  Hashtbl.iter (fun c i -> if i < size then idx2ch.(i) <- c) tbl;
  let ch2idx c = match Hashtbl.find_opt tbl c with Some i -> i | None -> 0 in
  let encode s = s |> String.to_seq |> List.of_seq |> List.map ch2idx in
  let decode ids =
    ids |> List.map (fun i -> idx2ch.(i)) |> List.to_seq |> String.of_seq
  in
  { size; ch2idx; encode; decode }

(* Build a dataset of (context, next) pairs using the high-level API. *)
let make_dataset ~block_size ~batch_size ~(v : vocab) names =
  let tokenize s = v.encode ("." ^ s ^ ".") in
  Dataset.sliding_window ~block_size ~tokenize ~device:c names
  |> Dataset.batch batch_size
  |> Dataset.shuffle ~buffer_size:200

(* Deterministic 90/10 split for evaluation. *)
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

(* 02 - Models Training
   --------------------------------------------------------------- *)

let make_optimizer ~lr ~weight_decay =
  if weight_decay > 0.0 then Optimizer.adamw ~lr ~weight_decay ()
  else Optimizer.adam ~lr ()

(* Compose a small MLP over a fixed context window. *)
let build_mlp ~vocab_size ~block_size ~n_embd ~n_hidden : Layer.module_ =
  let open Layer in
  sequential
    [
      embedding ~vocab_size ~embed_dim:n_embd ();
      flatten ();
      linear ~in_features:(block_size * n_embd) ~out_features:n_hidden ();
      tanh ();
      linear ~in_features:n_hidden ~out_features:vocab_size ();
    ]

let train ~model ~train_data ~val_data ~epochs ~lr ~weight_decay =
  let optimizer = make_optimizer ~lr ~weight_decay in
  let state, _history =
    Kaun.Training.fit ~model ~optimizer
      ~loss_fn:Loss.softmax_cross_entropy_with_indices ~train_data ~val_data
      ~epochs ~progress:true ~rngs:(Rng.key 42) ~device:c ~dtype:float32 ()
  in
  state

let eval_nll ~state ~val_data =
  let nll, _ =
    Kaun.Training.evaluate ~state ~dataset:val_data
      ~loss_fn:Loss.softmax_cross_entropy_with_indices ()
  in
  nll

(* 03 - Sampling
   -------------------------------------------------------------- *)

(* Build a model_fn compatible with Saga.Sampler for an MLP over a fixed
   context. It pads on the left with '.' to fill the context window. *)
let model_fn_of_mlp ~model ~params ~(v : vocab) ~block_size (tokens : int list)
    : float array =
  let open Array in
  let arr = of_list tokens in
  let pad = v.ch2idx '.' in
  let ctx = Array.make block_size pad in
  let take = Stdlib.min (length arr) block_size in
  blit arr (Stdlib.max 0 (length arr - take)) ctx (block_size - take) take;
  let input =
    Rune.create c float32 [| 1; block_size |] (Array.map float_of_int ctx)
  in
  model.Layer.apply params ~training:false input |> to_array

let print_generated_names ~model_fn ~(v : vocab) ~num ~max_new =
  let tokenizer (s : string) = v.encode s in
  let decoder ids = v.decode ids in
  let config =
    Saga.Sampler.(
      default |> with_do_sample true |> with_temperature 0.9
      |> with_max_new_tokens max_new)
  in
  let stop_on_eos =
    Saga.Sampler.eos_token_criteria ~eos_token_ids:[ v.ch2idx '.' ]
  in
  Printf.printf "\n[mlp] --- Generated names ---\n%!";
  for _i = 1 to num do
    let s =
      Saga.Sampler.generate_text ~model:model_fn ~tokenizer ~decoder ~prompt:"."
        ~generation_config:config ~stopping_criteria:[ stop_on_eos ] ()
    in
    (* Keep text between dots, if present *)
    let cleaned =
      match String.index_opt s '.' with
      | Some i -> (
          match String.index_from_opt s (i + 1) '.' with
          | Some j when j > i + 1 -> String.sub s (i + 1) (j - i - 1)
          | _ -> s)
      | None -> s
    in
    let len = String.length cleaned in
    let start = if len > 0 && cleaned.[0] = '.' then 1 else 0 in
    let stop =
      let l = String.length cleaned in
      if l > start && cleaned.[l - 1] = '.' then l - 1 else l
    in
    let cleaned =
      if stop > start then String.sub cleaned start (stop - start) else ""
    in
    if String.length cleaned > 0 then Printf.printf " - %s\n%!" cleaned
  done

(* 00 - Main
   ----------------------------------------------------------------- *)

let () =
  (* Load data + vocab *)
  let names = load_names "names.txt" in
  let v = build_vocab names in
  let train_names, eval_names = split_names_for_eval names in

  (* Dataset *)
  let block_size = 16 and batch_size = 256 in
  let train_ds = make_dataset ~block_size ~batch_size ~v train_names in
  let eval_ds = make_dataset ~block_size ~batch_size ~v eval_names in

  (* Model *)
  let model =
    build_mlp ~vocab_size:v.size ~block_size ~n_embd:64 ~n_hidden:64
  in

  (* Train + evaluate *)
  let state =
    train ~model ~train_data:train_ds ~val_data:eval_ds ~epochs:1 ~lr:2e-3
      ~weight_decay:0.01
  in
  Dataset.reset eval_ds;
  let nll = eval_nll ~state ~val_data:eval_ds in
  Printf.printf "[mlp] Eval NLL: %.4f nats, PPL: %.2f\n%!" nll (Stdlib.exp nll);

  (* Sample names *)
  let model_fn =
    model_fn_of_mlp ~model ~params:state.Training.State.params ~v ~block_size
  in
  print_generated_names ~model_fn ~v ~num:10 ~max_new:30
