from flask import Blueprint, render_template, url_for, flash, redirect, request, abort
from flask_login import current_user, login_required
from app import db
from app.models.item import Item
from app.utils.forms import ItemForm

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
@main_bp.route('/home')
def home():
    items = Item.query.order_by(Item.created_at.desc()).all()
    return render_template('home.html', title='Home', items=items)

@main_bp.route('/about')
def about():
    return render_template('about.html', title='About')

@main_bp.route('/item/new', methods=['GET', 'POST'])
@login_required
def new_item():
    form = ItemForm()
    if form.validate_on_submit():
        item = Item(
            title=form.title.data,
            description=form.description.data,
            author=current_user
        )
        db.session.add(item)
        db.session.commit()
        flash('Your item has been created!', 'success')
        return redirect(url_for('main.home'))
    
    return render_template('create_item.html', title='New Item', form=form, legend='New Item')

@main_bp.route('/item/<int:item_id>')
def item(item_id):
    item = Item.query.get_or_404(item_id)
    return render_template('item.html', title=item.title, item=item)

@main_bp.route('/item/<int:item_id>/update', methods=['GET', 'POST'])
@login_required
def update_item(item_id):
    item = Item.query.get_or_404(item_id)
    
    # Check if the current user is the author of the item
    if item.author != current_user:
        abort(403)
    
    form = ItemForm()
    
    if form.validate_on_submit():
        item.title = form.title.data
        item.description = form.description.data
        db.session.commit()
        flash('Your item has been updated!', 'success')
        return redirect(url_for('main.item', item_id=item.id))
    
    elif request.method == 'GET':
        form.title.data = item.title
        form.description.data = item.description
    
    return render_template('create_item.html', title='Update Item', form=form, legend='Update Item')

@main_bp.route('/item/<int:item_id>/delete', methods=['POST'])
@login_required
def delete_item(item_id):
    item = Item.query.get_or_404(item_id)
    
    # Check if the current user is the author of the item
    if item.author != current_user:
        abort(403)
    
    db.session.delete(item)
    db.session.commit()
    flash('Your item has been deleted!', 'success')
    return redirect(url_for('main.home'))

@main_bp.route('/user/<string:username>')
def user_items(username):
    from app.models.user import User
    user = User.query.filter_by(username=username).first_or_404()
    items = Item.query.filter_by(author=user).order_by(Item.created_at.desc()).all()
    return render_template('user_items.html', title=f"{username}'s Items", items=items, user=user)
