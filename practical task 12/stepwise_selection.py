import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import ModelDesc as md
from patsy import Term, EvalFactor

def update_model(model, formula):
    """
    Update a regression model with a new formula.

    :param model: a regression model from 'statsmodels' (OLS or GLM).
    :param formula: a string containing the statsmodels regression formula.
    
    :returns: the updated statsmodels regression model.
    """
    if isinstance(model, sm.OLS):
        return smf.ols(formula=formula, data=model.data.frame)
    elif isinstance(model, sm.GLM):
        return smf.glm(formula=formula, data=model.data.frame, family=model.family)    
        
def scope_formula_parser(formula, scope={"lower": None, "upper": None}):
    """
    Parse formula and search scope into ModelDesc variables used in for- and backward model selection.

    :param scope: a dictionary containing "lower" and "upper" search scope as statsmodel formulae.
    :param formula: a string containing the statsmodels regression formula.
    
    :returns: a list of ModelDesc containing backward, forward, and formula models.
    """
    # Default behavior for empty scope
    if scope["upper"] is None:
        scope.update({"upper": formula})
    if not scope["lower"]:
        scope.update({"lower": formula.split(" ")[0] + " ~ 1"})
        
    # Compute update factors (including interactions) using ModelDesc
    md_formula = md.from_formula(formula)

    md_forward = md.from_formula(scope["upper"])
    md_forward.rhs_termlist = list(set(md_forward.rhs_termlist)^set(md_formula.rhs_termlist)) 
    
    md_backward = md.from_formula(scope["lower"])
    md_backward.rhs_termlist = list(set(md_backward.rhs_termlist)^set(md_formula.rhs_termlist))
    
    # Remove from md_backward the main factors of interactions already in formula
    for t in md_formula.rhs_termlist:
        if len(t.factors) > 1:
            for f in t.factors:
                t_tmp = [Term([EvalFactor(f.name())])]
                md_backward.rhs_termlist = list(set(md_backward.rhs_termlist) - set(t_tmp))
                
    return md_backward, md_forward, md_formula
    
def forward_step(model, md_variables, criterion="aic"):
    """
    Perform all possible single forward steps for model selection.
    
    :param model: a regression model from 'statsmodels' (OLS or GLM).
    :param md_variables: a patsy.ModelDesc containing factors and interactions to test.
    :param criterion: a string indicating the information criterion used to select the best model ("aic" or "bic").
    
    :returns: a list of tuples containing the score for each candidate insertion.
    """
    scores_with_candidates = list()
    for variable in md_variables.rhs_termlist:
        candidate = " + " + variable.name() 
        tmp_model = update_model(model, model.formula + candidate)
        m_fit = tmp_model.fit()
        if criterion=="aic":
            scores_with_candidates.append((candidate, m_fit.aic))
        elif criterion=="bic":
            scores_with_candidates.append((candidate, m_fit.bic))
        
    return scores_with_candidates
            
def backward_step(model, md_variables, criterion="aic"):
    """
    Perform all possible single backward steps for model selection.
    
    :param model: a regression model from 'statsmodels' (OLS or GLM).
    :param md_variables: a patsy.ModelDesc containing factors and interactions to test.
    :param criterion: a string indicating the information criterion used to select the best model ("aic" or "bic").
    
    :returns: a list of tuples containing the score for each candidate deletion.
    """
    scores_with_candidates = list()
    for variable in md_variables.rhs_termlist:
        candidate = " - " + variable.name()  
        tmp_model = update_model(model, model.formula + candidate)
        m_fit = tmp_model.fit()
        if criterion=="aic":
            scores_with_candidates.append((candidate, m_fit.aic))
        elif criterion=="bic":
            scores_with_candidates.append((candidate, m_fit.bic))
            
    return scores_with_candidates

def stepwise_selection(model, scope={"lower": None, "upper": None}, direction="both", criterion="aic"):
    """
    Linear model designed by forward and backward selection.

    :param model: a regression model from 'statsmodels' (OLS or GLM). Not yet fitted!
    :param scope: a dictionary containing "lower" and "upper" search scope as statsmodel formulae.
    :param direction: a string indicating the direction of the single steps ("forward", "backward", or "both").
    :param criterion: a string indicating the information criterion used to select the best model ("aic" or "bic").

    :returns: an "optimal" fitted statsmodels regression model.
    """
    # Parse scope to get variables
    md_backward_vars, md_forward_vars, md_formula = scope_formula_parser(formula=model.formula, scope=scope)
    
    # Forward-Backward single-step changes
    current_score, best_new_score = 1e10, 1e10
    best_candidate = scope["lower"]
    all_md_vars = md_backward_vars.rhs_termlist + md_forward_vars.rhs_termlist
    while all_md_vars and best_candidate is not "":
        scores_with_candidates = list()
        model = update_model(model, md_formula.describe())
        m_fit = model.fit()
        print("Step:  " + criterion + "=", m_fit.aic)
        scores_with_candidates.append(("", m_fit.aic))
        # Forward
        if direction=="forward" or direction=="both":
            scores_with_candidates += forward_step(model=model, md_variables=md_forward_vars, criterion=criterion)    
        # Backward
        if direction=="backward" or direction=="both":
            scores_with_candidates += backward_step(model=model, md_variables=md_backward_vars, criterion=criterion)
            
        scores_with_candidates.sort(key=lambda score: score[1])
        print(*scores_with_candidates, sep = "\n")
        
        best_candidate, best_new_score = scores_with_candidates[0]
        if current_score > best_new_score:
            # Update
            formula_str = md_formula.describe() + best_candidate
            md_backward_vars, md_forward_vars, md_formula = scope_formula_parser(formula=formula_str, scope=scope)
            current_score = best_new_score
       
        all_md_vars = md_backward_vars.rhs_termlist + md_forward_vars.rhs_termlist
    
    best_model = update_model(model, md_formula.describe())     
    
    print("Result: " + criterion + "=", best_model.fit().aic)
    print(md_formula.describe())
    
    return best_model
