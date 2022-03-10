

# Argument parsing
def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def define_pdcnet_parser(subparsers):

    PDCNet = subparsers.add_parser('PDCNet', help='inference parameters for PDCNet')
    PDCNet.add_argument(
        '--confidence_map_R', default=1.0, type=float,
        help='R used for confidence map computation',
    )
    PDCNet.add_argument(
        '--multi_stage_type', default='direct', type=str, choices=['direct', 'd', 'D',
                                                                   'homography_from_last_level_uncertainty',
                                                                   'homography_from_quarter_resolution_uncertainty', 'h', 'H',
                                                                   'multiscale_homo_from_quarter_resolution_uncertainty', 'ms', 'MS'],
        help='multi stage type',
    )
    PDCNet.add_argument(
        '--ransac_thresh', default=1.0, type=float,
        help='ransac threshold used for multi-stages alignment',
    )
    PDCNet.add_argument(
        '--mask_type', default='proba_interval_1_above_5', type=str,
        help='mask computation for multi-stage alignment',
    )
    PDCNet.add_argument(
        '--homography_visibility_mask', default=True, type=boolean_string,
        help='apply homography visibility mask for multi-stage computation ?',
    )
    PDCNet.add_argument('--scaling_factors', type=float, nargs='+', default=[0.5, 0.6, 0.88, 1, 1.33, 1.66, 2],
                        help='scaling factors')
    PDCNet.add_argument(
        '--compute_cyclic_consistency_error', default=False, type=boolean_string,
        help='compute cyclic consistency error as additional uncertainty measure? Default is False',
    )


def define_model_parser(parser):
    """
    Parser for model selection and PDC-Net parameter selections, common to many scripts.
    """
    parser.add_argument('--model', type=str, required=True,
                        help='Model to use')
    parser.add_argument('--flipping_condition', default=False, type=boolean_string,
                        help='Apply flipping condition for semantic data and GLU-Net-based networks? ')
    parser.add_argument('--optim_iter', type=int, default=3,
                        help='number of optim iter for global GOCor, when applicable')
    parser.add_argument('--local_optim_iter', dest='local_optim_iter', type=int, default=None,
                        help='number of optim iter for local GOCor, when applicable')
    parser.add_argument('--path_to_pre_trained_models', type=str, default='pre_trained_models/',
                        help='path to the folder containing the pre trained model weights, or '
                             'path to the model checkpoint.')
    # add subparser for model types
    subparsers = parser.add_subparsers(dest='network_type')
    define_pdcnet_parser(subparsers)
