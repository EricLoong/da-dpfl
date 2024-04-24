def add_common_args(parser):
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument(
        "--num_workers", type=int, default=1, help="number of workers in data loader"
    )
    parser.add_argument("--tqdm", action="store_true")

    return parser
