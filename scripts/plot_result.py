import matplotlib
matplotlib.use("Agg")

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pygmt
from pathlib import Path

def plot_results(test: bool = False):
    """
    绘制并保存 Moho 反演结果的所有图件。

    参数:
        test (bool): 
            True  → 使用 test_run_approach.pkl，图片保存到 test/ 文件夹
            False → 使用 run_approach.pkl，图片保存到 fig/ 文件夹（默认）
    """
    # ===== 确定数据文件路径和图片保存目录 =====
    PROJECT_ROOT = Path.cwd().parent
    if test:
        pkl_file = PROJECT_ROOT/"result/test_run_approach.pkl"
        save_dir = PROJECT_ROOT/"test"
    else:
        pkl_file = PROJECT_ROOT/"result/run_approach.pkl"
        save_dir = PROJECT_ROOT/"fig"

    # 创建保存目录（已存在则不重复创建）
    os.makedirs(save_dir, exist_ok=True)

    # ===== 读取 pickle 数据 =====
    with open(pkl_file, "rb") as f:
        obj = pickle.load(f, encoding="latin1")

    # 提取所需变量并转为 numpy array
    density = np.asarray(obj['densities'])
    reference_levels = np.asarray(obj['reference_levels'])
    score_refden = np.asarray(obj['scores_refden'])
    score_regul = np.asarray(obj['scores_regul'])
    regul = np.asarray(obj['regul_params'])
    regul_residual = np.asarray(obj['regul_residuals'])
    refden_residual = np.asarray(obj['refden_residuals'])

    refden_moho_grid = obj['best_solutions_refden_moho_grid']
    refden_predict_grid = obj['best_solutions_refden_predict_grid']
    observe = obj['observe']
    lon = obj['lon']
    lat = obj['lat']
    lon_sub = obj['lon_sub']
    lat_sub = obj['lat_sub']

    print("数据形状：")
    print(np.shape(density))
    print(np.shape(reference_levels))
    print(np.shape(score_refden))

    # ====================== 1. Score vs Density & Reference Level ======================
    print('1. Score vs Density & Reference Level')
    fig, ax = plt.subplots(figsize=(7, 4.5))
    pcm = ax.pcolormesh(
        reference_levels, density, score_refden,
        shading='auto', cmap='viridis'
    )
    ax.set_xlabel('Reference level (km)', fontsize=12)
    ax.set_ylabel('Density contrast (kg m$^{-3}$)', fontsize=12)
    ax.tick_params(labelsize=11)

    cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
    cbar.set_label('Score', fontsize=12)
    cbar.ax.tick_params(labelsize=11)
    print('finished')

    # 找最小值并标注
    imin = np.unravel_index(np.nanargmin(score_refden), score_refden.shape)
    i_den, i_ref = imin
    den_best = density[i_den]
    ref_best = reference_levels[i_ref]
    score_best = score_refden[i_den, i_ref]

    ax.plot(ref_best, den_best, 'r*', markersize=12, markeredgecolor='k',
            zorder=10, label='Minimum score')

    ax.annotate(
        f'Min = {score_best:.2e}',
        xy=(ref_best, den_best),
        xycoords='data',
        xytext=(0.02, 0.95),
        textcoords='axes fraction',
        fontsize=10,
        ha='left', va='top',
        bbox=dict(boxstyle='round', fc='white', ec='0.7'),
        zorder=11
    )

    ax.legend(loc='lower left', frameon=True, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'score_density_reference_minimum.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('finished')

    # ====================== 2. Score vs Regularization Parameter ======================
    print('2. Score vs Regularization Parameter')
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(regul, score_regul, '-o', lw=1.5, ms=5, color='k')
    ax.set_xscale('log')
    ax.set_xlabel('Regularization parameter', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.tick_params(labelsize=11)

    i_min = np.nanargmin(score_regul)
    regul_best = regul[i_min]
    score_best = score_regul[i_min]

    ax.plot(regul_best, score_best, marker='*', color='red',
            markersize=12, markeredgecolor='k', zorder=10,
            label='Minimum score')

    ax.annotate(
        f'Min = {score_best:.2e}\n$\\lambda$ = {regul_best:.2e}',
        xy=(regul_best, score_best),
        xytext=(10, 10),
        textcoords='offset points',
        fontsize=10,
        bbox=dict(boxstyle='round', fc='white', ec='0.7'),
        arrowprops=dict(arrowstyle='->', lw=0.8)
    )

    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'score_vs_regularization.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('finished')

    # ====================== 3. Regularization Residual Histogram ======================
    print('3. Regularization Residual Histogram')
    fig, ax = plt.subplots(figsize=(6.5, 4))
    n_bins = 30
    ax.hist(regul_residual, bins=n_bins, color='skyblue',
            edgecolor='k', alpha=0.8)

    mean_val = np.mean(regul_residual)
    std_val = np.std(regul_residual)
    ax.axvline(mean_val, color='r', linestyle='--', lw=1.5,
               label=f'Mean = {mean_val:.2e}')
    ax.axvline(mean_val - std_val, color='g', linestyle=':', lw=1,
               label=f'-1σ')
    ax.axvline(mean_val + std_val, color='g', linestyle=':', lw=1,
               label=f'+1σ')

    ax.set_xlabel('Residual', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.tick_params(labelsize=11)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'regul_residual_histogram.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('finished')

    # ====================== 4. Refden Residual Histogram ======================
    print('4. Refden Residual Histogram')
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(refden_residual, bins=n_bins, color='skyblue',
            edgecolor='k', alpha=0.8)

    mean_val = np.mean(refden_residual)
    std_val = np.std(refden_residual)
    ax.axvline(mean_val, color='r', linestyle='--', lw=1.5,
               label=f'Mean = {mean_val:.2e}')
    ax.axvline(mean_val - std_val, color='g', linestyle=':', lw=1,
               label=f'-1σ')
    ax.axvline(mean_val + std_val, color='g', linestyle=':', lw=1,
               label=f'+1σ')

    ax.set_xlabel('Residual', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.tick_params(labelsize=11)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'refden_residual_histogram.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('finished')

    # ====================== 5. Moho Depth Map (PyGMT) ======================
    print('5. Moho Depth Map')
    nlat_moho, nlon_moho = refden_moho_grid.shape
    lon_moho = np.linspace(lon.min(), lon.max(), nlon_moho)
    lat_moho = np.linspace(lat.min(), lat.max(), nlat_moho)

    moho_grid = xr.DataArray(
        refden_moho_grid,  # 如果图东西颠倒或拉伸，改为 refden_moho_grid.T
        dims=("lat", "lon"),
        coords={"lat": lat_moho, "lon": lon_moho}
    )

    region = [lon_moho.min(), lon_moho.max(), lat_moho.min(), lat_moho.max()]

    fig = pygmt.Figure()
    fig.grdimage(grid=moho_grid, region=region, projection="R10c",
                frame=["af", "WSne"], cmap="jet")
    fig.colorbar(frame='af+lMoho depth (m)')
    fig.savefig(os.path.join(save_dir, "map_moho_depth.png"), dpi=300)
    print('finished')

    # ====================== 6. Predicted Gravity Map (PyGMT) ======================
    print('6. Predicted Gravity Map')
    nlat, nlon = refden_predict_grid.shape
    lon_grid = np.linspace(lon.min(), lon.max(), nlon)
    lat_grid = np.linspace(lat.min(), lat.max(), nlat)

    predict_grid = xr.DataArray(
        refden_predict_grid,
        dims=("lat", "lon"),
        coords={"lon": lon_grid, "lat": lat_grid}
    )
    region_predict = [lon_grid.min(), lon_grid.max(), lat_grid.min(), lat_grid.max()]

    fig = pygmt.Figure()
    fig.grdimage(grid=predict_grid, region=region_predict,
                 projection="R10c", cmap="viridis", frame=["af", "WSne"])
    fig.colorbar(frame='af+lPredicted gravity (mGal)')
    fig.text(x=lon_grid.mean(), y=lat_grid.max() + 0.5,
             text="Predicted", font="12p,Helvetica-Bold,black", justify="CT")
    fig.savefig(os.path.join(save_dir, "map_predict.png"), dpi=300)
    print('finished')

    # ====================== 7. Observed Gravity Map (PyGMT) ======================
    print('7. Observed Gravity Map')
    nlat_sub, nlon_sub = observe.shape
    lon_sub_grid = np.linspace(lon_sub.min(), lon_sub.max(), nlon_sub)
    lat_sub_grid = np.linspace(lat_sub.min(), lat_sub.max(), nlat_sub)

    observe_grid = xr.DataArray(
        observe,
        dims=("lat", "lon"),
        coords={"lon": lon_sub_grid, "lat": lat_sub_grid}
    )
    region_observe = [lon_sub_grid.min(), lon_sub_grid.max(),
                      lat_sub_grid.min(), lat_sub_grid.max()]

    fig = pygmt.Figure()
    fig.grdimage(grid=observe_grid, region=region_observe,
                 projection="R10c", cmap="viridis", frame=["af", "WSne"])
    fig.colorbar(frame='af+lObserved gravity (mGal)')
    fig.text(x=lon_sub_grid.mean(), y=lat_sub_grid.max() + 0.5,
             text="Observed", font="12p,Helvetica-Bold,black", justify="CT")
    fig.savefig(os.path.join(save_dir, "map_observe.png"), dpi=300)

    print(f"所有图件已成功保存至文件夹：{save_dir}")

import sys

# ===== 脚本入口 =====
if __name__ == "__main__":
    
    import os
    print("当前工作目录 =", os.getcwd())
    
    test = False

    if len(sys.argv) > 1 and sys.argv[1].lower() == "test":
        test = True

    plot_results(test=test)