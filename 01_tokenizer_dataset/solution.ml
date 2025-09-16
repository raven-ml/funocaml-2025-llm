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

let () =
  (* Sanity checks: vocab roundtrip and dataset window count *)
  let names = load_names "names.txt" in
  let vocab_size, _ch2idx, encode, decode = build_vocab names in
  let sample = "anna" in
  let ids = encode sample in
  let round = decode ids in
  let ok_round = String.length sample = String.length round in
  Printf.printf "[01] Vocab size: %d\n%!" vocab_size;
  Printf.printf "[01] Encode/Decode roundtrip length match: %b\n%!" ok_round;
  let block_size = 4 and batch_size = 16 in
  let ds = make_dataset ~block_size ~batch_size ~vocab_size ~encode names in
  (* Expected windows: sum over (|name|+1) because of wrapping with '.' both
     ends minus one next-token step *)
  let expected =
    List.fold_left (fun acc n -> acc + String.length n + 1) 0 names
  in
  Printf.printf "[01] Dataset created (approx windows ~= %d).\n%!" expected;
  ignore ds
