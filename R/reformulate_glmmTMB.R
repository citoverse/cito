# This file was taken and adjusted from glmmTMB https://github.com/glmmTMB/glmmTMB
.valid_covstruct <- c(
  e = 0
)


## backward compat (copied from lme4)
if ((getRversion()) < "3.2.1") {
  lengths <- function (x, use.names = TRUE) vapply(x, length, 1L, USE.NAMES = use.names)
}

if (getRversion() < "4.0.0") {
  deparse1 <- function (expr, collapse = " ", width.cutoff = 500L, ...) {
    paste(deparse(expr, width.cutoff, ...), collapse = collapse)
  }
}


expandDoubleVert <- function(term) {
  frml <- stats::formula(substitute(~x,list(x=term[[2]])))
  ## need term.labels not all.vars to capture interactions too:
  tt <- stats::terms(frml)
  newtrms <- lapply(attr(tt, "term.labels"),
                    function(t) {
                      sumTerms(list(0, toLang(t)))
                    })
  if(attr(tt, "intercept") != 0) {
    newtrms <- c(1, newtrms)
  }
  res <- lapply(newtrms,
                function(t) {
                  makeOp(
                    makeOp(t, term[[3]], quote(`|`)),
                    quote(`(`)
                  )
                })
  return(res)
}


RHSForm <- function(form, as.form=FALSE) {
  if (!as.form) return(form[[length(form)]])
  if (length(form)==2) return(form)  ## already RHS-only
  ## by operating on RHS in situ rather than making a new formula
  ## object, we avoid messing up existing attributes/environments etc.
  form[[2]] <- NULL
  ## assumes response is *first* variable (I think this is safe ...)
  if (length(vars <- attr(form,"variables"))>0) {
    attr(form,"variables") <- vars[-2]
  }
  if (is.null(attr(form,"response"))) {
    attr(form,"response") <- 0
  }
  if (length(facs <- attr(form,"factors"))>0) {
    attr(form,"factors") <- facs[-1,]
  }
  return(form)
}


`RHSForm<-` <- function(formula,value) {
  formula[[length(formula)]] <- value
  formula
}

#' combine a list of formula terms as a sum
#' @param termList a list of formula terms
sumTerms <- function(termList) {
  Reduce(function(x,y) makeOp(x,y,op=quote(`+`)),termList)
}


reOnly <- function(f, response=FALSE,bracket=TRUE) {
  flen <- length(f)
  f2 <- f[[2]]
  if (bracket)
    f <- lapply(lme4::findbars(f), makeOp, quote(`(`)) ## bracket-protect terms
  f <- sumTerms(f)
  if (response && flen==3) {
    form <- makeOp(f2, f, quote(`~`))
  } else {
    form <- makeOp(f, quote(`~`))
  }
  return(form)
}


makeOp <- function(x, y, op=NULL) {
  if (is.null(op) || missing(y)) {  ## unary
    if (is.null(op)) {
      substitute(OP(X),list(X=x,OP=y))
    } else {
      substitute(OP(X),list(X=x,OP=op))
    }
  } else substitute(OP(X,Y), list(X=x,OP=op,Y=y))
}

addForm0 <- function(f1,f2) {
  tilde <- as.symbol("~")
  if (!identical(utils::head(f2),tilde)) {
    f2 <- makeOp(f2,tilde)
  }
  if (length(f2)==3) warning("discarding LHS of second argument")
  RHSForm(f1) <- makeOp(RHSForm(f1),RHSForm(f2),quote(`+`))
  return(f1)
}

addForm <- function(...) {
  Reduce(addForm0,list(...))
}

#' list of specials -- taken from enum.R
findReTrmClasses <- function() {
  c(names(.valid_covstruct), "s")
}

toLang <- function(x) parse(text=x)[[1]]

expandGrpVar <- function(f) {
  form <- stats::as.formula(makeOp(f,quote(`~`)))
  mm <- stats::terms(form)
  tl <- attr(mm,"term.labels")
  ## reverse order: f/g -> f + g:f (for lme4/back-compatibility)
  switch_order <- function(x) paste(rev(unlist(strsplit(x, ":"))), collapse = ":")
  if (inForm(f, quote(`/`))) {
    ## vapply adds names; remove them, and reverse order of sub-terms, for back-compatibility ...
    tl <- unname(vapply(tl, switch_order, character(1)))
    tl <- rev(tl)
  }
  res <- lapply(tl, toLang)
  return(res)
}


expandAllGrpVar <- function(bb) {
  ## Return the list of expanded terms (/, *, ?)
  if (!is.list(bb))
    expandAllGrpVar(list(bb))
  else {
    for (i in seq_along(bb)) {
      esfun <- function(x) {
        if (length(x)==1 || !anySpecial(x, "|")) return(x)
        if (length(x)==2) {
          ## unary operator such as diag(1|f/g)
          ## return diag(...) + diag(...) + ...
          return(lapply(esfun(x[[2]]),
                        makeOp, y=utils::head(x)))
        }
        if (length(x)==3) {
          ## binary operator
          if (x[[1]]==quote(`|`)) {
            return(lapply(expandGrpVar(x[[3]]),
                          makeOp, x=x[[2]], op=quote(`|`)))
          } else {
            return(x)
            ## return(x) would be nice, but in that case x gets evaluated
            ## return(setNames(makeOp(esfun(x[[2]]), esfun(x[[3]]),
            ##  op=x[[1]]), names(x)))
          }
        }
      } ## esfun def.
      return(unlist(lapply(bb,esfun)))
    } ## loop over bb
  }
}

head.formula <- head.call <- function(x, ...) {
  x[[1]]
}

head.name <- function(x, ...) { x }

findbars_x <- function(term,
                       debug=FALSE,
                       specials=character(0),
                       default.special="us",
                       target = '|',
                       expand_doublevert_method = c("diag_special", "split")) {

  expand_doublevert_method <- match.arg(expand_doublevert_method)

  ds <- if (is.null(default.special)) {
    NULL
  } else {
    ## convert default special char to symbol (less ugly way?)
    eval(substitute(as.name(foo),list(foo=default.special)))
  }

  ## base function
  ## defining internally in this way makes debugging slightly
  ## harder, but (1) allows easy propagation of the top-level
  ## arguments down the recursive chain; (2) allows the top-level
  ## expandAllGrpVar() operation (which also handles cases where
  ## a naked term rather than a list is returned)

  fbx <- function(term) {
    if (is.name(term) || !is.language(term)) return(NULL)
    if (list(term[[1]]) %in% lapply(specials,as.name)) {
      if (debug) cat("special: ",deparse(term),"\n")
      return(term)
    }
    if (utils::head(term) == as.name(target)) {  ## found x | g
      if (debug) {
        tt <- if (target == '|') "bar" else sprintf('"%s"', target)
        cat(sprintf("%s term: %s\n", tt, deparse(term)))
      }
      if (is.null(ds)) return(term)
      return(makeOp(term, ds))
    }
    if (utils::head(term) == as.name("||")) {
      if (expand_doublevert_method == "diag_special") {
        return(makeOp(makeOp(term[[2]], term[[3]],
                             op = quote(`|`)),
                      as.name("diag")))
      }
      if (expand_doublevert_method == "split") {
        ## need to return *multiple* elements
        return(lapply(expandDoubleVert(term), fbx))
      }
      stop("unknown doublevert method ", expand_doublevert_method)
    }
    if (utils::head(term) == as.name("(")) {  ## found (...)
      if (debug) cat("paren term:",deparse(term),"\n")
      return(fbx(term[[2]]))
    }
    stopifnot(is.call(term))
    if (length(term) == 2) {
      ## unary operator, decompose argument
      if (debug) cat("unary operator:",deparse(term[[2]]),"\n")
      return(fbx(term[[2]]))
    }
    ## binary operator, decompose both arguments
    f2 <- fbx(term[[2]])
    f3 <- fbx(term[[3]])

    if (debug) { cat("binary operator:",deparse(term[[2]]),",",
                     deparse(term[[3]]),"\n")
      cat("term 2: ", deparse(f2), "\n")
      cat("term 3: ", deparse(f3), "\n")
    }
    c(f2, f3)
  }

  fbx_term <- fbx(term)
  if (debug) cat("fbx(term): ", deparse(fbx_term))
  expandAllGrpVar(fbx_term)

}

splitForm <- function(formula,
                      defaultTerm="us",
                      allowFixedOnly=TRUE,
                      allowNoSpecials=TRUE,
                      debug=FALSE,
                      specials = findReTrmClasses()) {

  ## logic:

  ## string for error message *if* specials not allowed
  ## (probably package-specific)
  noSpecialsAlt <- "lmer or glmer"

  ## formula <- expandDoubleVerts(formula)
  ## split formula into separate
  ## random effects terms
  ## (including special terms)

  fbxx <- findbars_x(formula, debug, specials)
  formSplits <- expandAllGrpVar(fbxx)

  if (length(formSplits)>0) {
    formSplitID <- sapply(lapply(formSplits, "[[", 1), as.character)
    # warn about terms without a
    # setReTrm method

    ## FIXME:: do we need all of this??

    if (FALSE) {
      badTrms <- formSplitID == "|"
      ## if(any(badTrms)) {
      ## stop("can't find setReTrm method(s)\n",
      ## "use findReTrmClasses() for available methods")
      ## FIXME: coerce bad terms to default as attempted below
      ## warning(paste("can't find setReTrm method(s) for term number(s)",
      ## paste(which(badTrms), collapse = ", "),
      ## "\ntreating those terms as unstructured"))
      formSplitID[badTrms] <- "("
      fixBadTrm <- function(formSplit) {
        makeOp(formSplit[[1]],quote(`(`))
        ## as.formula(paste(c("~(", as.character(formSplit)[c(2, 1, 3)], ")"),
        ## collapse = " "))[[2]]
      }
      formSplits[badTrms] <- lapply(formSplits[badTrms], fixBadTrm)

    }  ## skipped

    parenTerm <- formSplitID == "("
    # capture additional arguments
    reTrmAddArgs <- lapply(formSplits, "[", -2)[!parenTerm]
    # remove these additional
    # arguments
    formSplits <- lapply(formSplits, "[", 1:2)
    # standard RE terms
    formSplitStan <- formSplits[parenTerm]
    # structured RE terms
    formSplitSpec <- formSplits[!parenTerm]

    if (!allowNoSpecials) {
      if(length(formSplitSpec) == 0) stop(
        "no special covariance structures. ",
        "please use ",noSpecialsAlt,
        " or use findReTrmClasses() for available structures.")
    }

    reTrmFormulas <- c(lapply(formSplitStan, "[[", 2),
                       lapply(formSplitSpec, "[[", 2))
    reTrmFormulas <- unlist(reTrmFormulas) # Fix me:: added for rr structure when it has n = 2, gives a list of list... quick fix
    reTrmClasses <- c(rep(defaultTerm, length(formSplitStan)),
                      sapply(lapply(formSplitSpec, "[[", 1), as.character))
  } else {
    reTrmFormulas <- reTrmAddArgs <- reTrmClasses <- NULL
  }

  ## nobars() will get rid of any *naked* RE terms
  ## FIXME ... let noSpecials handle naked bar-terms if desired ?
  ## (would adding "|" to reTrmClasses work?)
  fixedFormula <- noSpecials(lme4::nobars(formula))

  list(fixedFormula  = fixedFormula,
       reTrmFormulas = reTrmFormulas,
       reTrmAddArgs  = reTrmAddArgs,
       reTrmClasses  = reTrmClasses)
}


noSpecials <- function(term, delete=TRUE, debug=FALSE, specials = findReTrmClasses()) {
  nospec <- noSpecials_(term, delete=delete, debug=debug, specials = specials)
  if (inherits(term, "formula") && length(term) == 3 && is.symbol(nospec)) {
    ## called with two-sided RE-only formula:
    ##    construct response~1 formula
    stats::as.formula(substitute(R~1,list(R=nospec)),
               env=environment(term))
    ## FIXME::better 'nothing left' handling
  } else if (is.null(nospec)) {
    ~1
  } else {
    nospec
  }
}

noSpecials_ <- function(term, delete=TRUE, debug=FALSE, specials = findReTrmClasses()) {
  if (debug) print(term)
  if (!anySpecial(term, specials)) return(term)
  if (length(term)==1) return(term)  ## 'naked' specials
  if (isSpecial(term, specials)) {
    if(delete) {
      return(NULL)
    } else { ## careful to return  (1|f) and not  1|f:
      return(substitute((TERM), list(TERM = term[[2]])))
    }
  } else {
    if (debug) print("not special")
    nb2 <- noSpecials_(term[[2]], delete=delete, debug=debug, specials = specials)
    nb3 <- if (length(term)==3) {
      noSpecials_(term[[3]], delete=delete, debug=debug, specials = specials)
    } else NULL
    if (is.null(nb2)) {
      return(nb3)
    } else if (is.null(nb3)) {
      if (length(term)==2 && identical(term[[1]], quote(`~`))) { ## special case for one-sided formula
        term[[2]] <- nb2
        return(term)
      } else {
        return(nb2)
      }
    } else {  ## neither term completely disappears
      term[[2]] <- nb2
      term[[3]] <- nb3
      return(term)
    }
  }
}

isSpecial <- function(term, specials = findReTrmClasses()) {
  if(is.call(term)) {
    ## %in% doesn't work (requires vector args)
    for(cls in specials) {
      if(term[[1]] == cls) return(TRUE)
    }
  }
  FALSE
}

isAnyArgSpecial <- function(term, specials = findReTrmClasses()) {
  for(tt in term)
    if(isSpecial(tt, specials)) return(TRUE)
  FALSE
}

anySpecial <- function(term, specials=findReTrmClasses()) {
  any(specials %in% all.names(term))
}

inForm <- function(form, value) {
  if (any(sapply(form,identical,value))) return(TRUE)
  if (all(sapply(form,length)==1)) return(FALSE)
  return(any(vapply(form,inForm,value,FUN.VALUE=logical(1))))
}


extractForm <- function(term,value) {
  if (!inForm(term,value)) return(NULL)
  if (is.name(term) || !is.language(term)) return(NULL)
  if (identical(utils::head(term),value)) {
    return(list(term))
  }
  if (length(term) == 2) {
    return(extractForm(term[[2]],value))
  }
  return(c(extractForm(term[[2]],value),
           extractForm(term[[3]],value)))
}

drophead <- function(term,value) {
  if (!inForm(term,value)) return(term)
  if (is.name(term) || !is.language(term)) return(term)
  if (identical(utils::head(term),value)) {
    return(term[[2]])
  }
  if (length(term) == 2) {
    return(drophead(term[[2]],value))
  } else  if (length(term) == 3) {
    term[[2]] <- drophead(term[[2]],value)
    term[[3]] <- drophead(term[[3]],value)
    return(term)
  } else stop("length(term)>3")
}



drop.special <- function(x, value=quote(offset), preserve = NULL) {
  k <- 0
  proc <- function(x) {
    if (length(x) == 1) return(x)
    if (x[[1]] == value && !((k <<- k+1) %in% preserve)) return(x[[1]])
    replace(x, -1, lapply(x[-1], proc))
  }
  ## handle 1- and 2-sided formulas
  if (length(x)==2) {
    newform <- substitute(~ . -x, list(x=value))
  } else {
    newform <- substitute(. ~ . - x, list(x=value))
  }
  return(stats::update(proc(x), newform))
}


replaceForm <- function(term,target,repl) {
  if (identical(term,target)) return(repl)
  if (!inForm(term,target)) return(term)
  if (length(term) == 2) {
    return(substitute(OP(x),list(OP=replaceForm(term[[1]],target,repl),
                                 x=replaceForm(term[[2]],target,repl))))
  }
  return(substitute(OP(x,y),list(OP=replaceForm(term[[1]],target,repl),
                                 x=replaceForm(term[[2]],target,repl),
                                 y=replaceForm(term[[3]],target,repl))))
}

no_specials <- function(term, specials = c("|", "||", "s")) {
  if (is.list(term)) {
    return(lapply(term, no_specials))
  }
  for (ss in specials) {
    if (identical(utils::head(term), as.name(ss))) return(term)
  }
  if (length(term) == 3) stop("don't know what to do")
  return(no_specials(term[[2]], specials))
}


sub_specials <- function (term,
                          specials = c("|", "||", "s"),
                          keep_args = c(2L, 2L, NA_integer_)) {
  if (is.name(term) || !is.language(term))
    return(term)
  ## previous version recursed immediately for unary operators,
  ## (we were only interested in `|`(x,y) and `||`(x,y))
  ## but here s(x) needs to be processed ...
  for (i in seq_along(specials)) {
    if (is.call(term) && term[[1]] == as.name(specials[i])) {
      if (is.na(keep_args[i])) {
        ## keep only *unnamed* args
        if (!is.null(names(term))) {
          term <- term[names(term)==""]
        }
      } else {
        term <- term[1:(1+keep_args[i])]
      }
      term[[1]] <- as.name("+")
      ## converts s(x) to +x, which is ugly, but
      ##  formula can handle repeated '+'
      ## discard additional arguments (e.g for s(x, ...))
      ## (fragile re: order??)
    }
  }
  for (j in 2:length(term)) {
    term[[j]] <- sub_specials(term[[j]],
                              specials = specials,
                              keep_args = keep_args)
  }
  term
}
