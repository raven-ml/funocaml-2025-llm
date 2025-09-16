(lang dune 3.19)

(name funocaml_2025_llm)

(source
 (github raven-ml/funocaml-2025-llm))

(pin
 (url git+https://github.com/raven-ml/raven)
 (package
  (name nx)))

(pin
 (url git+https://github.com/raven-ml/raven)
 (package
  (name nx-datasets)))

(pin
 (url git+https://github.com/raven-ml/raven)
 (package
  (name rune)))

(pin
 (url git+https://github.com/raven-ml/raven)
 (package
  (name saga)))

(pin
 (url git+https://github.com/raven-ml/raven)
 (package
  (name kaun)))

(package
 (allow_empty)
 (name funocaml_2025_llm)
 (synopsis "Workshop materials for FunOCaml 2025 LLM session")
 (description "A hands-on workshop introducing Raven to build LLM in OCaml")
 (depends
  (ocaml
   (>= 5.2.0))
  dune
  yojson
  nx
  rune
  saga
  kaun))
