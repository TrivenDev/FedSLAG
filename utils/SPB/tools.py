
def get_p_layers(super_type):
    if super_type == 'unlabeled' or super_type == 'point':
        return ['outc']
    elif super_type == 'scribble':
        return ['outc']
    elif super_type == 'bbox' or super_type == 'labeled':
        return ['outc', 'up4', 'up3', 'up2', 'up1']


def calculate_g_weight(data_ps, epoch, best_epoch, max_weight=0.9):
    distance_from_best = epoch - best_epoch
    fdata_weight = min(1.0 - data_ps, 1.0)
    distance_weight = min(distance_from_best / 10.0, 1.0)
    combined_weight = (fdata_weight + distance_weight) / 2.0
    g_weight = min(combined_weight, max_weight)
    g_weight = max(g_weight, 0.2)
    return g_weight


# 广播模型
def fed_broadcast(clients, nets, epoch, best_epoch, super_types, data_ps, step2_flag=False, fed_name='global'):
    for order, client in enumerate(clients):
        clt_super_type = super_types[order]
        clt_p_layers = get_p_layers(clt_super_type)
        g_weight = calculate_g_weight(data_ps[order], epoch, best_epoch)

        local_tmp_dict = nets[client].to('cuda:0').state_dict()
        g_model_dict = nets[fed_name].to('cuda:0').state_dict()

        param_keys = list(g_model_dict.keys())

        for idx, key in enumerate(param_keys):
            if not step2_flag:
                local_tmp_dict[key] = g_model_dict[key]
            else:  # step2
                if key in clt_p_layers:
                    local_tmp_dict[key] = (1.0 - g_weight) * g_model_dict[key] + (g_weight) * local_tmp_dict[key]
                else:
                    local_tmp_dict[key] = g_model_dict[key]
        nets[client].load_state_dict(local_tmp_dict)
