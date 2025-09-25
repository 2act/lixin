function plotXlsxCurves(filename)
% plotXlsxCurves 读取指定的 xlsx 文件并绘制曲线
%   第一列作为横坐标，后续各列作为纵坐标
%   第一行作为表头并用作图例
%
%   用法:
%   plotXlsxCurves('data.xlsx')

    % 读取表格（包含表头）
    tbl = readtable(filename);

    % 提取横坐标和纵坐标数据
    x = tbl{:,1};            % 第一列作为横坐标
    y = tbl{:,2:end};        % 其余列作为纵坐标
    
    % 创建图形
    figure; 
    hold on; grid on;
    
    % 绘制每一列曲线
    plot(x, y, 'LineWidth', 1.5);
    
    % 设置图例（使用表头）
    lgd = legend(tbl.Properties.VariableNames(2:end), 'Interpreter','none');
    set(lgd,'ItemHitFcn',@(~,evt) toggleVisibility(evt));
    
    function toggleVisibility(evt)
        if strcmp(evt.Peer.Visible,'on')
            evt.Peer.Visible = 'off';
        else
            evt.Peer.Visible = 'on';
        end
    end
    
    % 设置坐标标签
    xlabel(tbl.Properties.VariableNames{1}, 'Interpreter','none');
    ylabel('Values');
    title(['Curves from ', filename], 'Interpreter','none');
end
