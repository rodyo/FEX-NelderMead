%% Internal Nelder-Mead algorithm

% This algorithm is exactly the same as MATLAB's FMINSEARCH(), except
% that it has several improvements with regard to efficiency; see
%
% Singer & Singer, Applied Numerical Analysis & Computational Mathematics
% Volume 1 Issue 2, Pages 524 - 534,
% "Efficient Implementation of the Nelder-Mead Search Algorithm"
%
% for details.
function [sol, fval, exitflag, output] = NelderMead(funfcn, x0, options, varargin)

    % Please report bugs and inquiries to:
    %
    % Name    : Rody P.S. Oldenhuis
    % E-mail  : oldenhuis@gmail.com
    % Licence : 2-clause BSD (See License.txt)

    % If you find this work useful, please consider a donation:
    % https://www.paypal.me/RodyO/3.5


    % Check inputs
    error(nargchk(2,inf,nargin,'struct'))
    if ~isa(funfcn, 'function_handle')
        error('NelderMead:invalid_function',...
            'Input argument FUNFCN must be a valid function handle.');
    end

    if (nargin == 2) || isempty(options)
        options = optimset; end


    % Nelder-Mead algorithm control factors
    alpha = 1;     beta  = 0.5;
    gamma = 2;     delta = 0.5;
    a     = 1/20; % (a) is the size of the initial simplex.
    % 1/20 is 5% of the initial values.

    % constants
    N                         = numel(x0);
    originalSize              = size(x0);
    x0                        = x0(:);
    VSfactor_reflect          = alpha^(1/N);
    VSfactor_expand           = (alpha*gamma)^(1/N);
    VSfactor_inside_contract  = beta^(1/N);
    VSfactor_outside_contract = (alpha*beta)^(1/N);
    VSfactor_shrink           = delta;

    % Parse options
    reltol_x = optimget(options, 'TolX'  , 1e-4);
    reltol_f = optimget(options, 'TolFun', 1e-4);
    max_evaluations = optimget(options, 'MaxFunEvals', 200*N);
    max_iterations  = optimget(options, 'MaxIter', 1e4);
    display = optimget(options, 'display', 'off');

    % Check for output functions
    have_Outputfcn = false;
    if ~isempty(options.OutputFcn)
        have_Outputfcn = true;
        OutputFcn = options.OutputFcn;
    end

    % initial values
    iterations   = -1;   sort_inds = (1:N).';
    operation    = 0;    op = 'initial simplex';
    volume_ratio = 1;
    stop = false;

    % Generate initial simplex
    p = a/N/sqrt(2) * (sqrt(N+1) + N - 1) *  eye(N);
    q = a/N/sqrt(2) * (sqrt(N+1) - 1)     * ~eye(N);
    x = x0(:, ones(1,N));
    x = [x0, x + p + q];

    % function is known to be ``vectorized''
    f = funfcn(reshape(x, originalSize));
    evaluations = N+1;

    % first evaluate output function
    if have_Outputfcn
        optimValues.iteration = iterations;
        optimValues.funcCount = evaluations;
        optimValues.fval      = f(1);
        optimValues.procedure = op;
        OutputFcn(x0, optimValues, 'init', varargin{:});
    end

    % sort and re-label initial simplex
    [f, inds] = sort(f);  x = x(:, inds);

    % compute initial centroid
    C = sum(x(:, 1:end-1), 2)/N;

    % display header if per-iteration display is selected
    if strcmpi(display, 'iter')
        fprintf(1, ['\n\t\tf(1)\t\tfunc. evals.\toperation\n', ...
            '\t==================================================\n']);
    end

    % main loop
    while true

        % evaluate output function
        if have_Outputfcn
            optimValues.iteration  = iterations;
            optimValues.funcCount  = evaluations;
            optimValues.procedure  = op;
            [optimValues.fval,ind] = min(f);
            stop = OutputFcn(x(:,ind), optimValues, 'iter', varargin{:});
            if stop, break, end
        end

        % increase number of iterations
        iterations = iterations + 1;

        % display string for per-iteration display
        if strcmpi(display, 'iter')
            fprintf(1, '\t%+1.6e', f(1));
            fprintf(1, '\t\t%4.0d', evaluations);
            fprintf(1, '\t\t%s\n', op);
        end

        % re-sort function values
        x_replaced = x(:, end);
        if operation == 2  % shrink steps
            [f, inds] = sort(f);   x = x(:, inds);
        else   % non-shrink steps
            inds = f(end) <= f(1:end-1);
            f = [f(sort_inds(~inds)), f(end), f(sort_inds(inds))];
            x = [x(:, sort_inds(~inds)), x_replaced, x(:, sort_inds(inds))];
        end

        % update centroid (Singer & Singer are wrong here...
        % shrink & non-shrink steps should be treated the same)
        C = C + (x_replaced -  x(:, end))/N;

        % Algorithm termination conditions
        term_f = abs(f(end) - f(1)) / (abs(f(end)) + abs(f(1)))  < reltol_f;
        fail   = (iterations >= max_iterations) || ...
                 (evaluations >= max_evaluations);
        fail2  = all(~isfinite(f(:))) || all(~isfinite(x(:)));
        term_x = volume_ratio < reltol_x;
        if (term_x || term_f || fail || fail2), break, end

        % non-shrink steps are taken most of the time. Set this as the
        % default operation.
        operation = 1;

        % try to reflect the simplex
        xr = C + alpha*(C - x(:, end));
        fr = funfcn(xr);
        evaluations = evaluations + 1;

        % accept the reflection point
        if fr < f(end-1)
            x(:, end) = xr;
            f(end) = fr;

            % try to expand the simplex
            if fr < f(1)
                xe = C + gamma*(xr - C);
                fe = funfcn(xe);
                evaluations = evaluations + 1;

                % accept expand
                if (fe < f(1))
                    op = 'expand';
                    volume_ratio = VSfactor_expand * volume_ratio;
                    x(:, end) = xe;
                    f(end) = fe;
                    continue;
                end
            end

            % otherwise, just continue
            op = 'reflect';
            volume_ratio = VSfactor_reflect * volume_ratio;
            continue;

        % otherwise, try to contract the simplex
        else

            % outside contraction
            if fr < f(end)
                xc = C + beta*(xr - C);
                insouts = 1;

                % inside contraction
            else
                xc = C + beta*(x(:, end) - C);
                insouts = 2;
            end
            fc = funfcn(xc);
            evaluations = evaluations + 1;

            % accept contraction
            if fc < min(fr, f(end))
                switch insouts
                    case 1
                        op = 'outside contraction';
                        volume_ratio = VSfactor_outside_contract * volume_ratio;
                    case 2
                        op = 'inside contraction';
                        volume_ratio = VSfactor_inside_contract * volume_ratio;
                end
                x(:, end) = xc;
                f(end) = fc;
                continue;

            % everything else has failed - shrink the simplex towards x1
            else
                % first shrink
                operation = 2;
                xones = x(:, ones(1, N+1));
                x = xones + delta*(x - xones);
                f = funfcn(x);
                evaluations = evaluations + N + 1;
                volume_ratio = VSfactor_shrink * volume_ratio;
                % then evaluate output function
                op = 'shrink';
            end

        end % select next procedure
    end % main loop

    % evaluate output function
    if have_Outputfcn
        optimValues.iteration = iterations;
        optimValues.funcCount = evaluations;
        optimValues.fval      = f(1);
        optimValues.procedure = 'Final simplex.';
        OutputFcn(x(:,1), optimValues, 'done', varargin{:});
    end

    % final values
    sol  = x(:,1);
    fval = f(1);

    % exitflag
    if (term_x || term_f), exitflag = 1; end % normal convergence
    if stop, exitflag = -1; end              % stopped by outputfunction
    if fail, exitflag = 0; end               % max. iterations or max. func. eval. exceeded
    if fail2, exitflag = -3; end             % everything is non-finite

    % create output structure
    output.iterations = iterations;
    output.funcCount  = evaluations;
    switch exitflag
        case 1
            output.message = sprintf(['Optimization terminated:\n',...
                ' the current x satisfies the termination criteria using OPTIONS.TolX of %d \n',...
                ' and F(X) satisfies the convergence criteria using OPTIONS.TolFun of %d \n'],...
                reltol_x, reltol_f);
        case 0
            if (iterations >= max_iterations)
                output.message = sprintf(['Optimization terminated:\n',...
                    ' Maximum amount of iterations exceeded; \nIncrease ',...
                    '''MaxIters'' option.\n']);
            elseif (evaluations >= max_evaluations)
                output.message = sprintf(['Optimization terminated:\n',...
                    ' Maximum amount of function evaluations exceeded; \nIncrease ',...
                    '''MaxFunevals'' option.\n']);
            end
        case -1
            output.message = ...
                sprintf('Optimization terminated by user-provided output function.\n');
        case -3
            output.message = ...
                sprintf('All function values are non-finite. Exiting...\n');
    end

    % display convergence
    if ~isempty(options.Display) && ~strcmpi(options.Display, 'off')
        fprintf(1, '\n%s\n', output.message); end

    % make sure the algorithm is correct
    output.algorithm  = 'Nelder-Mead simplex direct search';

end % NelderMead
